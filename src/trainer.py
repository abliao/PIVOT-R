from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import os
import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import logging
from agent import Agent
from models.robotic_transformer_pytorch import KeyWorld
from feeders import Feeder
from utils import configure_optimizer, set_seed, LR_Scheduler

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"   
    os.environ["MASTER_PORT"] = "1097"  
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    

def lr_lambda(current_step):
    warmup_steps = 100  
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1/(((current_step+10000)/10000)**0.5)


class Trainer:
    def __init__(self, cfg: DictConfig, args, ngpus_per_node) -> None:
        if cfg['common']['seed'] is not None:
            set_seed(cfg['common']['seed'])

        self.cfg = cfg
        self.start_epoch = 1
        args.gpu = int(os.environ['LOCAL_RANK'])
        ddp_setup(args.gpu, ngpus_per_node)
        self.total_workers = ngpus_per_node
        self.gpu = args.gpu
        self.device = 'cuda:{}'.format(args.gpu) 
        self.ckpt_dir = args.output_path
        init_seed(self.cfg['common']['seed'])

        if self.cfg['training']['should']:
            feeder = Feeder(**cfg['datasets']['train'])
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(feeder, shuffle=True)
            self.train_dataset = DataLoader(
                dataset=feeder,
                batch_size=cfg['datasets']['batch_size'],
                num_workers=cfg['datasets']['num_worker'],
                drop_last=True,
                pin_memory=True,
                worker_init_fn=init_seed,
                sampler = self.train_sampler)

        if self.cfg['evaluation']['should']:
            feeder = Feeder(**cfg['datasets']['test'])
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(feeder, shuffle=False)
            self.test_dataset = DataLoader(
                dataset=feeder,
                batch_size=cfg['datasets']['batch_size'],
                num_workers=cfg['datasets']['num_worker'],
                drop_last=True,
                pin_memory=True,
                worker_init_fn=init_seed,
                sampler = self.test_sampler)
            
        assert self.cfg['training']['should'] or self.cfg['evaluation']['should']
        import yaml
        with open(self.cfg['model']) as f:
            model_cfg = yaml.safe_load(f)
        model = KeyWorld(**model_cfg).to(self.device)
        if cfg['common']['resume']:
            state_dict = torch.load(cfg['initialization']['path_to_checkpoint'], map_location=self.device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[6:13]=='module.':
                    name = k[13:] 
                else:
                    name = k[6:]
                new_state_dict[name] = v
            state_dict = new_state_dict
            model.load_state_dict(state_dict)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        self.agent = Agent(model=model,cfg=cfg)
        
        print(f'{sum(p.numel() for p in self.agent.parameters())} parameters in agent')
        try:
            print(f'{sum(p.numel() for p in self.agent.model.embedder.parameters())} parameters in agent.embedder')
        except:
            pass
        self.optimizer_agent = torch.optim.AdamW(self.agent.parameters(), lr=cfg['training']['agent']['learning_rate'], weight_decay=cfg['training']['agent']['weight_decay'])
        self.lr_scheduler_agent=torch.optim.lr_scheduler.LambdaLR(self.optimizer_agent, lr_lambda)

    def run(self) -> None:
        min_loss = None
        for epoch in range(self.start_epoch, 1 + self.cfg['common']['epochs']):
            if self.gpu == 0:
                print(f"\nEpoch {epoch} / {self.cfg['common']['epochs']}\n")
            start_time = time.time()
            to_log = []
            save_best_should = False
            
            self.train_sampler.set_epoch(epoch)
            if self.cfg['training']['should']:
                to_log += self.train_agent(epoch)

            if self.cfg['evaluation']['should'] and (epoch % self.cfg['evaluation']['every'] == 0):
                to_log += self.eval_agent(epoch)
                if min_loss is None or min_loss>to_log[-1]['agent/eval/total_loss']:
                    min_loss = to_log[-1]['agent/eval/total_loss']
                    save_best_should = True
                
            if self.gpu == 0 and self.cfg['training']['should']:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg['common']['do_checkpoint'], save_best_should=save_best_should)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            if self.gpu == 0:
                print(to_log)
        self.finish()
    
    def eval(self) -> None:
        Tester(self.agent,self.cfg,self.episode_dir)
        self.finish()
        
    def train_agent(self, epoch: int) -> None:
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor = {}, {}, {}

        cfg_agent = self.cfg['training']['agent']
        steps_per_epoch = len(self.train_dataset)

        if epoch > cfg_agent['start_after_epochs']:
            metrics_agent = self.train_component(self.agent, self.optimizer_agent, steps_per_epoch=steps_per_epoch, lr_scheduler=self.lr_scheduler_agent, **cfg_agent)
        self.agent.eval()

        return [{'epoch': epoch, **metrics_agent}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int,  max_grad_norm: Optional[float],  lr_scheduler= None, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)
        mean_loss = 0
        torch.autograd.set_detect_anomaly(True)
        if self.gpu == 0:
            for _, (imgs, instr, actions, states_tensor, next_imgs, index) in enumerate(tqdm(self.train_dataset, desc="Training", ncols=100)):
                """batch['observation'] is supposed to be channels first and in [0, 1]"""
                B, F, C, H, W = imgs.shape
                imgs = imgs.contiguous().view(B, F, C, H, W).float()
                next_imgs = next_imgs.contiguous().view(B, C, H, W).float()
                _, V = actions.shape
                B, F, V2 = states_tensor.shape
                actions = actions.contiguous().view(-1, V).float()
                states_tensor = states_tensor.contiguous().view(-1, F, V2).float()
                next_imgs = next_imgs.unsqueeze(dim=1)
                instructions = [] 
                for i in instr:
                    instructions += [i]
                batch=dict()
                batch['observations']=imgs
                batch['next_observations']=next_imgs
                batch['states']=states_tensor
                batch['actions']=actions
                batch['instr']=instructions
                optimizer.zero_grad()
                batch = self._to_device(batch)
                losses = component.compute_loss(batch, **kwargs_loss) 
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                loss_total_epoch += loss_total_step.item() / steps_per_epoch

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
        else:
            for _, (imgs, instr, actions, states_tensor, next_imgs, index) in enumerate(self.train_dataset):
                """batch['observation'] is supposed to be channels first and in [0, 1]"""
                B, F, C, H, W = imgs.shape
                imgs = imgs.contiguous().view(B, F, C, H, W).float()
                next_imgs = next_imgs.contiguous().view(B, C, H, W).float()
                _, V = actions.shape
                B, F, V2 = states_tensor.shape
                actions = actions.contiguous().view(-1, V).float()
                states_tensor = states_tensor.contiguous().view(-1, F, V2).float()
                next_imgs = next_imgs.unsqueeze(dim=1)
                instructions = [] 
                for i in instr:
                    instructions += [i]
                batch=dict()
                batch['observations']=imgs
                batch['next_observations']=next_imgs
                batch['states']=states_tensor
                batch['actions']=actions
                batch['instr']=instructions
                optimizer.zero_grad()
                batch = self._to_device(batch)
                losses = component.compute_loss(batch, **kwargs_loss) 
                loss_total_step = losses.loss_total
                loss_total_step.backward()
                loss_total_epoch += loss_total_step.item() / steps_per_epoch

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics
    

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model,metrics_actor = {}, {}, {}

        metrics_agent = self.eval_component(self.agent)

        return [metrics_agent]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        if self.gpu == 0:
            for _, (imgs, instr, actions, states_tensor, next_imgs, index) in enumerate(tqdm(self.test_dataset, desc="Testing", ncols=100)):
                """batch['observation'] is supposed to be channels first and in [0, 1]"""
                B, F, C, H, W = imgs.shape
                imgs = imgs.contiguous().view(B, F, C, H, W).float()
                next_imgs = next_imgs.contiguous().view(B, C, H, W).float()
                _, V = actions.shape
                B, F, V2 = states_tensor.shape
                actions = actions.contiguous().view(-1, V).float()
                states_tensor = states_tensor.contiguous().view(-1, F, V2).float()
                next_imgs = next_imgs.unsqueeze(dim=1)
                instructions = [] 
                for i in instr:
                    instructions += [i] 
                batch=dict()
                batch['observations']=imgs
                batch['next_observations']=next_imgs
                batch['actions']=actions
                batch['states']=states_tensor
                batch['instr']=instructions
                batch = self._to_device(batch)
                losses = component.compute_loss(batch, **kwargs_loss)
                loss_total_epoch += losses.loss_total.item()
                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value
                steps += 1
        else:
            for _, (imgs, instr, actions, states_tensor, next_imgs, index) in enumerate(self.test_dataset):
                """batch['observation'] is supposed to be channels first and in [0, 1]"""
                B, F, C, H, W = imgs.shape
                imgs = imgs.contiguous().view(B, F, C, H, W).float()
                next_imgs = next_imgs.contiguous().view(B, C, H, W).float()
                _, V = actions.shape
                B, F, V2 = states_tensor.shape
                actions = actions.contiguous().view(-1, V).float()
                states_tensor = states_tensor.contiguous().view(-1, F, V2).float()
                next_imgs = next_imgs.unsqueeze(dim=1)
                instructions = [] 
                for i in instr:
                    instructions += [i]
                batch=dict()
                batch['observations']=imgs
                batch['next_observations']=next_imgs
                batch['actions']=actions
                batch['states']=states_tensor
                batch['instr']=instructions
                batch = self._to_device(batch)
                losses = component.compute_loss(batch, **kwargs_loss)
                loss_total_epoch += losses.loss_total.item()
                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value
                steps += 1
        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics
    def _save_checkpoint(self, epoch: int, save_agent_only: bool, save_best_should) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir + '/last.pt')
        if save_best_should:
            torch.save(self.agent.state_dict(), self.ckpt_dir + '/best.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir + '/epoch.pt')
            torch.save({
                "optimizer_agent": self.optimizer_agent.state_dict(),
                "lr_scheduler_agent":self.lr_scheduler_agent.state_dict()
            }, self.ckpt_dir + '/optimizer.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool, save_best_should) -> None:
        self._save_checkpoint(epoch, save_agent_only,save_best_should)


    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) if torch.is_tensor(batch[k]) else batch[k] for k in batch}

    def finish(self) -> None:
        pass
