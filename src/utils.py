from collections import OrderedDict
import cv2
from pathlib import Path
import random
import os
import shutil
from PIL import Image
import glob
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

LR_Scheduler = {'StepLR':StepLR,
                'CosineAnnealingLR':CosineAnnealingLR}

def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Parameter)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames):
    assert frames.ndim == 4 # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()



def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence

def data_processing(sim,depth_mat,target_id,action_description,file_prefix,frame_id):
    target_name = sim.objs[sim.objs.ID==target_id].Name.values[0]

    ignore = np.array([0,128])
    unique_values = np.unique( depth_mat.ravel())
    # assert  (unique_values==ignore[0]).any() and (unique_values==ignore[1]).any(), 'mask may have been changed'
    unique_values = np.setdiff1d(unique_values,ignore)
    # assert (unique_values==sim.objs[sim.objs.ID==target_id].mask_id.values[0]).any(), f'mask may have been changed, target_name:{target_name}, unique_values:{unique_values}'
    mat_bool = depth_mat==sim.objs[sim.objs.ID==target_id].mask_id.values[0]
    mask = depth_mat.copy()
    mask[mat_bool]=255
    mask[~mat_bool]=0
    mask=mask[:,:,0]

    def mask_to_points(mask):
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_points = []
        
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            contour_points = [[float(point[0][0]),float(point[0][1])] for point in approx]
            contours_points.append(contour_points)
            
        return contours_points

    points_list = mask_to_points(mask)
    # assert len(points_list)<=1, 'too many objects'
    data={}
    data['text']=['Pick a '+target_name+'.']
    data['is_sentence']=True
    data['shapes']=[]
    data['action_description'] = action_description
    
    for points in points_list:
        shape={"label": "target",
        "labels": [
            "target"
        ],
        "shape_type": "polygon",
        "image_name": file_prefix+f"/{frame_id:03}.jpg",
        "points":points,
        "group_id": None,
        "group_ids": [
            None
        ],
        "flags": {}
        }
        data['shapes'].append(shape)
    return data

def get_normalized_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # normalized bbox
    height, width = mask.shape
    y_min_normalized = y_min / height
    y_max_normalized = y_max / height
    x_min_normalized = x_min / width
    x_max_normalized = x_max / width
    
    return np.array((y_min_normalized, x_min_normalized, y_max_normalized, x_max_normalized))

def select_instr(sample,objs,instructions,level):
    random.seed(42)
    x,y,z=sample['robot_location']
    if 'event' not in sample.keys():
        event = 'graspTargetObj'
    else:
        event = sample['event']
    targetObjID=sample['targetObjID']
    target = objs[objs.ID == sample['targetObjID']].iloc[0]
    targetObj = objs[objs.ID==targetObjID].Name.values[0]
    other_id = []
    for obj in sample['objList'][:]:
        if obj[0]!=targetObjID:
            other_id.append(obj[0])
    other = objs[objs.ID.isin(other_id)]
    if target.Name not in instructions[event].keys():
        level=0
    final_instrs = []
    if level >1:
        instrs = instructions[event][target.Name]
        for way in instrs.keys():
            instr = instrs[way]
            if way=='descriptions':
                can_att = ['name', 'shape', 'application', 'other']
                if target.Name in other.Name.values:
                    can_att.remove('name')
                if target.Shape in other.Shape.values:
                    can_att.remove('shape')
                if target.Application in other.Application.values:
                    can_att.remove('application')
                if target.Other in other.Other.values:
                    can_att.remove('other')    
                if len(sample['objList'])>1 and (target.Size > other.Size.values+1).all():
                    can_att.append('largest')
                if len(sample['objList'])>1 and (target.Size < other.Size.values-1).all():
                    can_att.append('smallest')
            else:
                origin_att = ['left','right','close','distant','left front','front right','behind left','behind rght']
                target_index = sample['target_obj_index']-1
                loc1 = sample['objList'][target_index][1:3]
                if len(sample['objList'])>1:
                    for obj in sample['objList'][:]:
                        if obj[0]==sample['targetObjID']:
                            continue
                        loc2 = obj[1:3]
                        can_att = []
                        if loc1[1]-loc2[1]>5:
                            can_att.append('left')
                        if loc1[1]-loc2[1]<-5:
                            can_att.append('right')
                        if loc1[0]-loc2[0]>5:
                            can_att.append('close')
                        if loc1[0]-loc2[0]<-5:
                            can_att.append('distant')   
                        if loc1[1]-loc2[1]>5 and loc1[0]-loc2[0]<-5:
                            can_att.append('left front') 
                        if loc1[1]-loc2[1]<-5 and loc1[0]-loc2[0]<-5:
                            can_att.append('front right') 
                        if loc1[1]-loc2[1]>5 and loc1[0]-loc2[0]>5:
                            can_att.append('behind left')     
                        if loc1[1]-loc2[1]<-5 and loc1[0]-loc2[0]>5:
                            can_att.append('behind rght')  
                        origin_att = set(origin_att).intersection(set(can_att))
                        origin_att = list(origin_att)
                    can_att = origin_att
                else:
                    can_att = []
            
            have_att = set(instr.keys())
            can_att = list(set(can_att).intersection(have_att))
            if len(can_att)==0:
                selected_instr = []
            else:
                selected_instr = []
                for att in can_att:
                    if 'origin' in  instr[att].keys():
                        selected_instr.append(instr[att]['origin'])
                    if 'human' in instr[att].keys():
                        selected_instr+=instr[att]['human']
                    
            final_instrs += selected_instr

    if level==0 or len(final_instrs)==0:
        if event == 'graspTargetObj':
            final_instrs = ['Pick a '+targetObj+'.']
        elif event == 'placeTargetObj':
            final_instrs = ['Place ' + targetObj+'.']
        elif event == 'moveNear':
            otherObj = other.Name.values[0]
            final_instrs = [f'Move {targetObj} near {otherObj}.']
        elif event == 'knockOver':
            final_instrs = ['Knock ' + targetObj +' over'+'.']
        elif event == 'pushFront':
            final_instrs = ['Push ' + targetObj + ' front'+'.']
        elif event == 'pushLeft':
            final_instrs = ['Push ' + targetObj + ' left'+'.']
        elif event == 'pushRight':
            final_instrs = ['Push ' + targetObj + ' right'+'.']
    return random.choice(final_instrs)