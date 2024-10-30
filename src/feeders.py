import os
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import math
import random
import grpc
import pandas as pd
import numpy as np
import re
from PIL import Image
import time
import json
from tqdm import tqdm
import sys
import cv2
from torchvision import transforms
from utils import *

try:
    from . import GrabSim_pb2_grpc
    from . import GrabSim_pb2
    from .Env.simUtils import SimServer
except:
    from Env import GrabSim_pb2_grpc
    from Env import GrabSim_pb2
    from Env.simUtils import SimServer

def Resize(mat,img_size=224):
    if isinstance(img_size,int):
        img_size = (img_size, img_size)
    if isinstance(mat,np.ndarray):
        if mat.dtype !=np.uint8:
            mat = (mat*255).astype(np.uint8)
        mat = Image.fromarray(mat, mode='RGB')
    mat = mat.resize(img_size)
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat/255.0
    return mat

def find_img(frame,img_size=256):
    if 'img'+str(img_size) in frame.keys():
        return frame['img'+str(img_size)]
    return Resize(frame['img'],img_size)

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
    action = anno['action_description']
    
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

    return mask, comments, is_sentence, action


actuatorRanges=np.array([[-30.00006675720215, 31.65018653869629],
 [-110.00215911865234, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-159.9984588623047, 129.99838256835938],
 [-15.000033378601074, 150.00035095214844],
 [-5.729577541351318, 64.74422454833984],
 [-30.00006675720215, 30.00006675720215],
 [-30.00006675720215, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-45.00010299682617, 58.49898910522461],
 [-39.999900817871094, 39.999900817871094],
 [-90.00020599365234, 90.00020599365234],
 [-45.00010299682617, 45.00010299682617],
 [-30.00006675720215, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-110.00215911865234, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-5.729577541351318, 64.74422454833984],
 [-129.99838256835938, 159.9984588623047],
 [-150.00035095214844, 15.000033378601074],
 [-5.729577541351318, 64.74422454833984],
 [-30.00006675720215, 30.00006675720215],
 [-30.00006675720215, 30.00006675720215],
 [-90.00020599365234, 90.00020599365234]])


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    def __call__(self, img):
        np_img = np.array(img)
        noise = np.random.normal(self.mean, self.std, np_img.shape).astype(np.float32)
        noisy_img = np_img + noise
        noisy_img = np.clip(noisy_img, 0, 255) 
        noisy_img = Image.fromarray(noisy_img.astype(np.uint8)) 
        return noisy_img

class Feeder(Dataset):
    objs = SimServer.objs
    def __init__(self, data_path, instructions_path, control='ee', history_len=3, instructions_level=[3] bin=256, img_size=224, data_size=None,dataAug=True,image_process=None):
        self.data_path = data_path
        self.instructions_path = instructions_path
        
        self.control = control
        self.instructions_level = instructions_level
        self.history_len = history_len
        self.bin = bin
        self.img_size = img_size
        self.dataAug = dataAug

        self.id2name = {0: 'Mug', 1: 'Banana', 2: 'Toothpaste', 3: 'Bread', 4: 'Softdrink',5: 'Yogurt',6: 'ADMilk',7: 'VacuumCup',8: 'Bernachon',9: 'BottledDrink',10: 'PencilVase',11: 'Teacup',12: 'Caddy',13: 'Dictionary',14: 'Cake',15: 'Date',16: 'Stapler',17: 'LunchBox',18: 'Bracelet',19: 'MilkDrink',20: 'CocountWater',21: 'Walnut',22: 'HamSausage',23: 'GlueStick',24: 'AdhensiveTape',25: 'Calculator',26: 'Chess',27: 'Orange',28: 'Glass',29: 'Washbowl',30: 'Durian',31: 'Gum',32: 'Towl',33: 'OrangeJuice',34: 'Cardcase',35: 'RubikCube',36: 'StickyNotes',37: 'NFCJuice',38: 'SpringWater',39: 'Apple',40: 'Coffee',41: 'Gauze',42: 'Mangosteen',43: 'SesameSeedCake',44: 'Glove',45: 'Mouse',46: 'Kettle',47: 'Atomize',48: 'Chips',49: 'SpongeGourd',50: 'Garlic',51: 'Potato',52: 'Tray',53: 'Hemomanometer',54: 'TennisBall',55: 'ToyDog',56: 'ToyBear',57: 'TeaTray',58: 'Sock',59: 'Scarf',60: 'ToiletPaper',61: 'Milk',62: 'Soap',63: 'Novel',64: 'Watermelon',65: 'Tomato',66: 'CleansingFoam',67: 'CocountMilk',68: 'SugarlessGum',69: 'MedicalAdhensiveTape',70: 'SourMilkDrink',71: 'PaperCup',72: 'Tissue'}
        self.name2id = {v: k for k, v in self.id2name.items()}

        self.read_data(self.data_path,self.instructions_path,self.instructions_level,data_size)
        import open_clip
        _, _, self.image_preprocess = open_clip.create_model_and_transforms(image_process['clip_arch'], pretrained=image_process['clip_path'])
        self.origin_image_preprocess = self.image_preprocess
        if dataAug:
            transform_list = list(self.image_preprocess.transforms)
            new_transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0)
            transform_list.insert(2, new_transform)
            updated_transform = transforms.Compose(transform_list)
            self.image_preprocess = updated_transform

    def read_data(self,data_paths,instructions_path,instructions_level,data_size=None):
        total_files=[]
        for path in data_paths:
            files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            files = sorted(files)
            total_files+=files
        if isinstance(data_size,int):
            total_files=total_files[:data_size]
        elif isinstance(data_size,float):
            data_size = int(len(total_files)*data_size)
            total_files=total_files[:data_size]
            
        with open(instructions_path,'rb') as f:
            instructions = pickle.load(f)
        self.data,self.instructions = total_files, instructions
        self.total_imgs = []
        self.total_states = []
        self.total_next_imgs = []
        self.total_actions = []
        self.total_instrs = []
        self.total_belongs = []
        for i,tra in tqdm(enumerate(total_files)):
            imgs,states,next_imgs,actions,instrs,belongs = self.get_data_from_one_tra(tra)
            self.total_imgs += imgs
            self.total_states += states
            self.total_next_imgs += next_imgs
            self.total_actions += actions
            self.total_instrs += instrs
            self.total_belongs += belongs

    def get_data_from_one_tra(self,tra):
        files = [os.path.join(tra, item) for item in os.listdir(tra) if os.path.isfile(os.path.join(tra, item)) and item.endswith('pkl')]
        assert len(files)==1
        file = files[0]
        with open(file,'rb') as f:
            sample=pickle.load(f)
        images = [os.path.join(tra, item) for item in os.listdir(tra) if os.path.isfile(os.path.join(tra, item)) and item.endswith('jpg')]
        images = sorted(images)
        jsons = [os.path.join(tra, item) for item in os.listdir(tra) if os.path.isfile(os.path.join(tra, item)) and item.endswith('json')]
        jsons = sorted(jsons)
        json_contents = []

        for file in jsons:
            with open(file) as f:
                data=json.load(f)
            json_contents.append(data)
        x,y,z=sample['robot_location']
        if 'event' not in sample.keys():
            event = 'graspTargetObj'
        else:
            event = sample['event']
        targetObjID=sample['targetObjID']
        targetObj = self.objs[self.objs.ID==targetObjID].Name.values[0]
        level=random.choice(self.instructions_level)
        target = self.objs[self.objs.ID == sample['targetObjID']].iloc[0]
        other_id = []
        for obj in sample['objList'][:]:
            if obj[0]!=targetObjID:
                other_id.append(obj[0])
        other = self.objs[self.objs.ID.isin(other_id)]
        if len(other)>0:
            otherObj = other.Name.values[0]
        if target.Name not in self.instructions.keys():
            level=0
        final_instrs = []
        if event == 'graspTargetObj':
            final_instrs = ['Pick a '+targetObj+'.']
        elif event == 'placeTargetObj':
            final_instrs = ['Place ' + targetObj+'.']
        elif event == 'moveNear':
            final_instrs = [f'Move {targetObj} near {otherObj}.']
        elif event == 'knockOver':
            final_instrs = ['Knock ' + targetObj +' over'+'.']
        elif event == 'pushFront':
            final_instrs = ['Push ' + targetObj + ' front'+'.']
        elif event == 'pushLeft':
            final_instrs = ['Push ' + targetObj + ' left'+'.']
        elif event == 'pushRight':
            final_instrs = ['Push ' + targetObj + ' right'+'.']
        if level >0:
            instrs = self.instructions[event][target.Name]
            for way in instrs.keys():
                instr = instrs[way]
                if way!='descriptions':
                    continue
                if way=='descriptions':
                    can_att = ['name', 'color', 'shape', 'application', 'other']
                    if target.Name in other.Name.values:
                        can_att.remove('name')
                    if target.Color in other.Color.values:
                        can_att.remove('color')
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
                        selected_instr.append(instr[att]['origin'])
                final_instrs += selected_instr

        imgs = []
        masks = []
        bounding_boxes = []
        action_descriptions = []
        states = []
        actions=[]
        next_imgs = []
        now_joints = [0]*14 + [36.0,-40.0,40.0,-90.0,5.0,0.0,0.0]
        last_action = np.array(sample['initLoc'])

        for frame_id,frame in enumerate(sample['trajectory'][:-1]):
            with Image.open(images[frame_id]) as f:
                img = f.resize((224, 224),Image.LANCZOS)
            imgs.append(img) 
            sensors=frame['state']['sensors']
            state = np.array(sensors[3]['data'])
            state[:3]-=np.array([x,y,z])
            state[:]/=np.array([50,30,40])
            states.append(state)

            if self.control == 'ee':
                if frame['action'][5]>=5:
                    frame['action'][5]=0
                if len(frame['action'])==6:
                    frame['action'] = [*frame['action'],0,0]
                if frame['action'][5]>=1:
                    frame['action'][5] = 1
                def discretize_value(value, num_bins=256):
                    value_clipped = np.clip(value, -1, 1)
                    discretized = np.round((value_clipped + 1) / 2 * (num_bins - 1))
                    return discretized
                frame['action'] = discretize_value(frame['action'])
                action = np.array(frame['action'], dtype=np.float64)
            else:
                before_joints = frame['state']['joints']
                before_joints = [joint['angle'] for joint in before_joints]
                after_joints = sample['trajectory'][frame_id+1]['state']['joints'] 
                after_joints = [joint['angle'] for joint in after_joints]
                map_id=[0,1,2,3,6,9,12,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,
                    33,36,39,42,43,44,46,47,48]
                before_joints=[before_joints[id] for id in map_id]
                after_joints=[after_joints[id] for id in map_id]
                joints = (np.array(after_joints)-np.array(before_joints)) /(actuatorRanges[:,1]-actuatorRanges[:,0])*50
                action = np.array([joints[-12],joints[-11],joints[-6],joints[-5],frame['action'][-1]],dtype=np.float64)
            actions.append(action)
            last_action = frame['action']
        
        with Image.open(images[-1]) as f:
            next_img = f.resize((224, 224),Image.LANCZOS)
        next_imgs = [next_img] * len(imgs)
        tmp_imgs=[]
        tmp_states=[]
        for i in range(len(imgs)):
            if i+1>=self.history_len:
                tmp_imgs.append(imgs[i-self.history_len+1:i+1])
                tmp_states.append(states[i-self.history_len+1:i+1])
            else:
                prefix = self.history_len-(i+1)
                tmp_imgs.append(imgs[:1]*prefix+imgs[:i+1])
                tmp_states.append(states[:1]*prefix+states[:i+1])
            tmp_states[-1]=np.array(tmp_states[-1])
        imgs=tmp_imgs
        states=tmp_states 
        
        final_instrs = [final_instrs]*len(imgs)
        belongs = [tra]*len(imgs)
        return imgs,states,next_imgs,actions,final_instrs,belongs

    def __len__(self):
        return len(self.total_imgs)

    def __iter__(self):
        return self

    def genObjwithLists(self, sim_client,sceneID,objList):
        for x,y,z,yaw,type in objList:
            obj_list = [GrabSim_pb2.ObjectList.Object(x=x, y=y, yaw=yaw, z=z, type=type)]
            scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=sceneID))
    
    def __getitem__(self, index):
        
        imgs,instr,actions,states,next_imgs = self.total_imgs[index],self.total_instrs[index],self.total_actions[index],self.total_states[index],self.total_next_imgs[index]
        instr = random.choice(instr)
        process_imgs = []
        for img in imgs:
            process_imgs.append(self.image_preprocess(img))
        try:
            next_imgs = self.origin_image_preprocess(next_imgs)
        except Exception as e:
            print('error', e)
            print('belongs',self.total_belongs[index])
            next_imgs = self.origin_image_preprocess(next_imgs)
        actions_tok = torch.from_numpy(actions)
        imgs_tensor = torch.stack(process_imgs, dim=0)
        states_tensor = torch.from_numpy(states)
        next_imgs_tensor = next_imgs
        return imgs_tensor, instr, actions_tok, states_tensor, next_imgs_tensor, index
    