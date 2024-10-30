import threading
import logging
from pathlib import Path
from typing import Any, Optional, Tuple
from PIL import Image
import io
import time
import base64
import requests
import json
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import pack, unpack, repeat, reduce, rearrange
from utils import extract_state_dict
from utils import init_weights, LossWithIntermediateLosses

from prompts.prompt_task import prompt_task
from prompts.task2actions import task2actions
from prompts.prompt_action_will_do import prompt_action_will_do

class Agent(nn.Module):
    def __init__(self, model,cfg):
        super().__init__()
        self.loss_weight = cfg['training']['agent']['loss_weight']
        self.model = model 
        self.batch = {}
        self._running = True
        self.buffer_lock = threading.Lock() 

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO) 
        import open_clip
        _, _, self.image_preprocess = open_clip.create_model_and_transforms(cfg['image_process']['clip_arch'], pretrained=cfg['image_process']['clip_path'])
    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def __repr__(self) -> str:
        return "agent"
    
    def reset(self):
        self._running = True
        self.batch = {}
    def load(self, path_to_checkpoint: Path, device: torch.device) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in agent_state_dict.items():
            name = k[:6]+k[13:]  
            new_state_dict[name] = v
        agent_state_dict = new_state_dict
        self.load_state_dict(agent_state_dict)

    def act(self, batch: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        with torch.no_grad():
            obs = batch['observations']
            states = batch['states']
            texts = batch['instr']
        predict_actions, prediction = self(obs,texts,states)
        return predict_actions

    def update_target_tokenizer(self):
        if self.tokenizer is None:
            return
        for model_param, shadow_param in zip(self.tokenizer.parameters(), self.target_tokenizer.parameters()):
            shadow_param.data = (1.0 - self.momentum) * shadow_param.data + self.momentum * model_param.data
        self.momentum = min(1., self.momentum+self.momentum_delta)

    def forward(self,obs,texts,states):
        actions,prediction = self.model(obs,texts,states)
        return actions, prediction
    
    def compute_loss(self, batch, **kwargs: Any) -> LossWithIntermediateLosses:

        obs = batch['observations']
        next_obs = batch['next_observations']
        states = batch['states']
        texts = batch['instr']
        actions = batch['actions']
        predict_actions, prediction = self(obs,texts,states)
        loss_actions = (F.cross_entropy(predict_actions.reshape(-1,256).float(), actions.reshape(-1,).long()) ) * self.loss_weight[0]
        loss_observation=loss_actions*0

        if prediction is not None:
            training=self.training
            self.eval()
            with torch.no_grad():
                next_token,text_token = self.model(next_obs,texts,states,return_embed=True)
            loss_observation = F.mse_loss(prediction, next_token)*self.loss_weight[1]
            
            if training:
                self.train()
        return LossWithIntermediateLosses(loss_actions=loss_actions,loss_observation=loss_observation)

    def update_input(self,batch):
        with self.buffer_lock:
            for k in batch.keys():
                if isinstance(batch[k],torch.Tensor):
                    self.batch[k] = batch[k].clone().detach()
                elif isinstance(batch[k],str):
                        self.batch[k] = batch[k]
                else:
                    self.batch[k] = batch[k].copy()
    
    def update_instr(self):
        while self.batch is None or self.batch =={}:
            time.sleep(0.1)
        with self.buffer_lock:
            batch = {}
            for k in self.batch.keys():
                if isinstance(self.batch[k],torch.Tensor):
                    batch[k] = self.batch[k].clone().detach()
                elif isinstance(self.batch[k],str):
                    batch[k] = self.batch[k]
                else:
                    batch[k] = self.batch[k].copy()

        url = 'http://127.0.0.1:8000/inference/'
        def pil_image_to_base64(image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG") 
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        def send_request(image,prompt):
            data = {
                'image': pil_image_to_base64(image),  
                'prompt': prompt
            }

            response = requests.post(url, json=data)

            if response.status_code == 200:
                answer=response.json()['response'][3:-4].strip()
                return answer
            else:
                assert False, f"Failed to get valid response, status code: {response.status_code}" 
                self.logger.debug(f"Failed to get valid response, status code: {response.status_code}")

        while True:
            try:
                image = batch['mat']
                task = batch['instr'][0]
                prompt = prompt_task.format(task=task)
                answer = send_request(image, prompt)
                answer = json.loads(answer)
                task_format = answer['task']
                actions = task2actions[task_format]
                break
            except Exception as e:
                self.logger.debug(f'error {e}')

        while self._running:
            try:
                with self.buffer_lock:
                    batch = {}
                    for k in self.batch.keys():
                        if isinstance(self.batch[k],torch.Tensor):
                            batch[k] = self.batch[k].clone().detach()
                        elif isinstance(self.batch[k],str):
                            batch[k] = self.batch[k]
                        else:
                            batch[k] = self.batch[k].copy()
                image = batch['mat']
                task = batch['instr'][0]
                prompt = prompt_action_will_do.format(task=task,actions=actions,hand_state=batch['hand_state'])
                answer = send_request(image, prompt)
                answer = answer.replace('\_', '_')
                answer = json.loads(answer)
                action = answer['action']
                with self.buffer_lock:
                    self.batch['new_instr'] = [task+f'; action: {action}']
                self.logger.debug(f'action {action}')
                time.sleep(0.1)
            except Exception as e:
                self.logger.debug(f'error {e}')


    def update_output(self):
        self.logger.debug('start update_output')
        while self.batch is None or self.batch =={} or 'new_instr' not in self.batch:
            time.sleep(0.01)
        with self.buffer_lock:
            batch = {}
            for k in self.batch.keys():
                if isinstance(self.batch[k],torch.Tensor):
                    batch[k] = self.batch[k].clone().detach()
                elif isinstance(self.batch[k],str):
                    batch[k] = self.batch[k]
                else:
                    batch[k] = self.batch[k].copy()
        self.logger.debug(f"new_instr {batch['new_instr']}")
        batch['instr'] = batch['new_instr']
        action = self.act(batch)
        return action
