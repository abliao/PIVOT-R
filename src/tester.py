import hydra
from omegaconf import DictConfig
import re
import yaml
from pathlib import Path
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
import pickle
import random
from copy import deepcopy
import time
import threading
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip, ImageClip, concatenate_videoclips

from google.protobuf import message
import grpc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from Env import GrabSim_pb2_grpc, GrabSim_pb2, initJointsArrange
from Env.simUtils import *
from utils import *
from agent import Agent
from models.robotic_transformer_pytorch import KeyWorld

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


def read_data(path):
    import re
    f=open(path)
    data=[]
    for line in f.readlines():
        line = line.strip('\n') 
        data.append(line)

    datas=[]
    last_index=0
    for i in range(len(data)):
        if data[i]=='':
            datas.append(data[last_index:i])
            last_index=i+1
    df=[]
    for i in datas:
        data=[]
        for j in i:
            result = re.split(',|;', j)
            numbers=list(map(float, result))
            data.append(numbers)
        df.append(data)
    return df

def action_untokenization(env, action,bins,joints_arrange):
    joints=action*(joints_arrange[-7:,1]-joints_arrange[-7:,0])/50
    return joints

def genObjwithLists(sim_client,sceneID,objList):
    for x,y,z,yaw,type in objList:
        obj_list = [GrabSim_pb2.ObjectList.Object(x=x, y=y, yaw=yaw, z=z, type=type)]
        scene = sim_client.MakeObjects(GrabSim_pb2.ObjectList(objects=obj_list, sceneID=sceneID))

def get_image(sim_client,sceneID):
    caremras=[GrabSim_pb2.CameraName.Head_Color]
    action = GrabSim_pb2.CameraList(sceneID=sceneID, cameras=caremras)
    im = sim_client.Capture(action).images[0]
    mat = np.frombuffer(im.data,dtype=im.dtype).reshape((im.height, im.width, im.channels))
    return mat

def get_depth(sim_client,sceneID):
    caremras=[GrabSim_pb2.CameraName.Head_Depth]
    action = GrabSim_pb2.CameraList(sceneID=sceneID, cameras=caremras)
    im = sim_client.Capture(action).images[0]
    mat = np.frombuffer(im.data,dtype=im.dtype).reshape((im.height, im.width, im.channels))
    t=100 #150
    mat = 1.0 * mat
    mat[mat>t]=t
    return mat
        
datas=[]

def is_element_in_string(element_list, target_string):
    for element in element_list:
        if element in target_string:
            return True
    return False

from PIL import Image
def Resize(mat):
    mat = Image.fromarray(mat, mode='RGB')
    mat = mat.resize((224,224)) 
    mat = np.array(mat)
    mat = 1.0 * mat
    mat = mat/255.0
    return mat

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_inpout(sim,agent,device,robot_location):
    mats = sim.getImage()
    mat = Image.fromarray(mats[0])
    obs = mat.resize((224, 224),Image.LANCZOS)
    obs = agent.image_preprocess(obs)
    img=torch.Tensor(obs)
    img=img.reshape(-1,1,*img.shape).to(device)
    sensors=sim.getState()['sensors']
    state = np.array(sensors[3]['data'])
    state[:3]-=robot_location
    state[:]/=np.array([50,30,40])
    state=torch.Tensor(state).to(device).unsqueeze(0).unsqueeze(0)
    return mat, img, state
def grasp(sim,agent,log,target_obj_index,robot_location,device='cuda',history_len=1,handSide='Right',control='joint',other_obj_index=None):
    check_fun = {
        'graspTargetObj':sim.checkGraspTargetObj,
        'placeTargetObj':sim.checkPlaceTargetObj,
        'moveNear':sim.checkMoveNear,
        'knockOver':sim.checkKnockOver,
        'pushFront':sim.checkPushFront,
        'pushLeft':sim.checkPushLeft,
        'pushRight':sim.checkPushRight
    }
    if not isinstance(robot_location, np.ndarray):
        robot_location=np.array(robot_location)
    instr=log['instruction']
    event=log['event']
    if event=='moveNear':
        max_steps=120
        objs_info = {'obj1_id':target_obj_index,'obj2_id':other_obj_index,}
    else:
        max_steps=80
        objs_info = {'obj_id':target_obj_index,}
    target_oringin_loc=sim.getObjsInfo()[1]['location']

    mat, img, state = get_inpout(sim,agent,device,robot_location)
    states=torch.repeat_interleave(state, history_len, dim=1)
    imgs=torch.repeat_interleave(img, history_len, dim=1)
    log['imgs']=[sim.getImage()[0]]
    joints = np.array(sim.getActuators())
    ready4grasp = 1
    agent.reset()
    VLM_task = threading.Thread(target=agent.update_instr)
    VLM_task.start()
    for _ in range(max_steps):
        time.sleep(0.03)
        now_instr = instr
        mat, img, state = get_inpout(sim,agent,device,robot_location)
        
        if history_len==1:
            imgs = img
            states = state
        else:
            imgs = torch.cat([imgs[:,-history_len+1:],img],dim=1)
            states = torch.cat([states[:,-history_len+1:],state],dim=1)
        assert imgs.shape[1]==history_len, f"length of input sequence is error, needed {history_len} not {imgs.shape[1]}"
        batch={}
        if sim.grasp_state[handSide]==0:
            batch['hand_state']='open'
        else:
            batch['hand_state']='closed'
        batch['mat']=mat
        batch['observations']=img
        batch['states']=state
        batch['instr']=[now_instr]
        agent.update_input(batch)
        predict = agent.update_output()
        predict=predict[0].cpu().detach().numpy()
        last_action=predict

        last_action = last_action.argmax(axis=1)
        last_action = last_action/(256-1)*2-1 
        last_action[-2:] = np.round(last_action[-2:])
        
        assert last_action[-1]==0 or last_action[-1]==1, print(f'gripper is {last_action[-1]}')
        if handSide=='Right':
            last_action[:3],last_action[3:6]= last_action[3:6], last_action[:3]
        else:
            last_action[-2],last_action[-1] = last_action[-1],last_action[-2]
        if control=='ee':
            if sim.grasp_state[handSide]==0:
                msg=sim.moveHand(x=last_action[0],y=last_action[1],z=last_action[2],keep_rpy=(0,0,0),method='diff',gap=0.1,handSide=handSide)
            else:
                msg=sim.moveHand(x=last_action[0],y=last_action[1],z=last_action[2],method='diff',gap=0.1,handSide=handSide)
        else:
            now_joints = np.array(sim.getActuators())
            joint_ids = [-12,-11,-6,-5]
            joints[joint_ids] = now_joints[joint_ids]
            last_action[:4] = last_action[:4]*(actuatorRanges[joint_ids,1]-actuatorRanges[joint_ids,0])/50
            joints[joint_ids] += last_action[:4]
            sim.changeJoints(joints)
        
        if_grasp = last_action[-1]==1
        if if_grasp:
            ready4grasp-=1
        if ready4grasp==0 and sim.grasp_state[handSide]==0:
            sim.grasp(angle=(65,68),handSide=handSide)
            log['grasp_img'] = sim.getImage()[0]
        elif not if_grasp and sim.grasp_state[handSide]==1:
            sim.release()
            ready4grasp=1
        
        log['track'].append(last_action.copy())
        log['imgs'].append(sim.getImage()[0])

        if check_fun[event](**objs_info):
            log['info']='success'
            break
        if _==max_steps-1:
            log['info']='time_exceed'
            break
    agent._running = False 
    VLM_task.join()
    return log

def Tester(agent, cfg, output_path):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    levels = cfg['datasets']['eval']['levels']
    client = cfg['env']['client']
    history_len = cfg['datasets']['history_len']
    action_nums = cfg['env']['num_actions']
    bins = cfg['env']['bins']
    mode = cfg['env']['mode']
    control = cfg['env']['control']
    max_steps = cfg['env']['max_steps']
    device = cfg['common']['device']
    agent.load(**cfg['initialization'], device=device)
    agent.to(device)
    agent.eval()

    scene_num = 1
    map_id = 2
    server = SimServer(client, scene_num=scene_num, map_id=map_id)
    sim = SimAction(client, scene_id=0)
    handSide = 'Right'

    with open(cfg['datasets']['test']['instructions_path'], 'rb') as f:
        instructions = pickle.load(f)
    directories = cfg['datasets']['eval']['data_path']
    
    logs = []
    def list_dirs(directory):
        files = []
        for entry in os.listdir(directory):
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path):
                files.append(full_path)
        return files
    def find_pkl(directory):
        files = [os.path.join(directory, item) for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)) and item.endswith('pkl')]
        assert len(files)==1
        return files[0]

    for level,directory in zip(levels,directories):
        episode_dir = output_path/str(level)
        episode_dir.mkdir(parents=True, exist_ok=True)
        if level>1:
            with open(cfg['datasets']['eval']['instr_path']+str(level)+'.pkl', 'rb') as f:
                instructions = pickle.load(f)
        dirs = list_dirs(directory)
        success = 0
        total_num = 0
        for index in tqdm(range(len(dirs))):

            logger.info(f"files_index: {dirs[index]}")
            sim.EnableEndPointCtrl(True)
            sim.reset()
            if control == 'joint':
                sim.EnableEndPointCtrl(False)
            else:
                sim.EnableEndPointCtrl(True)
            with open(find_pkl(dirs[index]), 'rb') as f:
                data = pickle.load(f)
            if 'event' not in data.keys():
                event = 'graspTargetObj'
            else:
                event = data['event']
            
            desk_id = data['deskInfo']['id']  
            sim.addDesk(desk_id=desk_id, h=98)
            can_list = list(SimServer.can_list)

            objList = data['objList'] 
            sim.addObjects(objList)
            target_obj_index = data['target_obj_index']
            obj_id = objList[target_obj_index - 1][0]
            target_obj_id = obj_id
            targetObj = sim.objs[sim.objs.ID == target_obj_id].Name.values[0]
            other_obj_index = None
            if event == 'moveNear':
                other_ids = [i for i in range(len(objList)) if i != target_obj_index-1]
                other_obj_index = other_ids[0]+1
                other_obj_id = objList[other_obj_index-1][0]
                otherObj = sim.objs[sim.objs.ID == other_obj_id].Name.values[0]
            action=0
            log = {}
            log['objs'] = objList
            log['deskInfo'] = {'desk_id': desk_id, 'height': sim.desk_height}
            log['detail'] = ''
            log['track'] = []
            log['targetObjID'] = target_obj_id
            log['targetObj'] = targetObj

            if level<=1:
                if event == 'graspTargetObj':
                    instr = 'Pick a '+targetObj+'.'
                elif event == 'placeTargetObj':
                    instr = 'Place ' + targetObj+'.'
                    
                elif event == 'moveNear':
                    instr = 'Move ' + targetObj+' near '+otherObj+'.'
                elif event == 'knockOver':
                    instr = 'Knock ' + targetObj +' over'+'.'
                elif event == 'pushFront':
                    instr = 'Push ' + targetObj + ' front'+'.'
                elif event == 'pushLeft':
                    instr = 'Push ' + targetObj + ' left'+'.'
                elif event == 'pushRight':
                    instr = 'Push ' + targetObj + ' right'+'.'
            else:
                instr = select_instr(data,sim.objs,instructions,level)

            log['instruction'] = instr
            log['event'] = event
            sx, sy = sim.getObservation().location.X, sim.getObservation().location.Y
            robot_location = (sx, sy, 90)
            if event == 'placeTargetObj':
                for action in sim.graspTargetObj(target_obj_index,distance=20):
                    pass
            log = grasp(sim, agent, log, target_obj_index=target_obj_index, robot_location=robot_location,
                        device=device, history_len=history_len, control=control, handSide=handSide,other_obj_index=other_obj_index)

            images = [ImageClip(frame.astype(np.uint8), duration=1 / 6) for frame in log['imgs']]
            clip_images = concatenate_videoclips(images)

            del log['imgs']
            logs.append(log)

            if log['info'] == 'success':
                success += 1

            total_num += 1
            logger.info(f'num: {total_num}, success rate:{success / total_num * 100:.2f}%)')
            time.sleep(1)
            if log['info'] in ['success', 'collision', 'time_exceed']:
                im = sim.getImage()[0]
                plt.imshow(im)
                plt.savefig(episode_dir / f"{index:04d}_{log['info']}_{event}_{log['targetObj']}.png", format='png')
                if 'grasp_img' in log.keys():
                    im = log['grasp_img']
                    plt.imshow(im)
                    plt.savefig(episode_dir / f"{index:04d}_grasp_{log['info']}_{event}_{log['targetObj']}.png", format='png')
                clip_images.write_videofile(str(episode_dir / f"{index:04d}_grasp_{log['info']}_{event}_{log['targetObj']}.mp4"), fps=6)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='test', type=str)
    args = parser.parse_args()
    with open('config/trainer.yaml') as f:
        cfg = yaml.safe_load(f)
    with open(cfg['model']) as f:
        model_cfg = yaml.safe_load(f)
    model = KeyWorld(**model_cfg).cuda()
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
    agent = Agent(model=model,cfg=cfg)
    output_path = Path(args.output_path)
    trainer = Tester(agent,cfg,output_path)