import grpc
import time
import random
import numpy as np
import pandas as pd
import math
import transforms3d as tf

try:
    from . import GrabSim_pb2, GrabSim_pb2_grpc
except:
    import GrabSim_pb2, GrabSim_pb2_grpc

class SimServer():

    desks=pd.DataFrame( [[0,'Desk',0,-70,0-5], 
            [1,'KitchenDesk',90,-140,0-5],
            [2,'WoodDesk',80,-70,0-5],
            # [3,'WoodDesk2',60, -70, 0-5],
            [4,'MetalDesk',81, -70, 0-5],
            [5,'CoffeeTable',40, -80, 0-5],
            [6,'OfficeDrawerDesk',57.5, -50, -10-5],
            [7,'KitchenDesk2',84.7, -60, 0-5],
            [8,'DiningTable',72.3, -90, 0-5],
            [9,'ReceptionTable',74.81, -60, 0-5],
            [10,'OfficeDesk',72.5, -110, 0-5],
            [11,'OfficeDesk2',80, -150, -10-5],
        ],columns=['ID','Name','H','X','Y'])
    data = {
    'ID': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    'Name': ['ADMilk', 'GlueStick', 'Bernachon', 'Cup', 'Yogurt', 'NFCJuice', 'Milk', 'CocountWater', 'Chips','bottle', 'cabinet', 'AnMuXi', 'QuChenShi', 'XingBaKe'],
    'center_height': [6.65707397460938, 2.6456069946289, 6.65229797363281, 4.74349212646484, 9.58502960205078, 6.60990142822266, 6.61824035644531, 6.62626647949219, 10.65509796142578, None, None, None, None, None],
    'Color': ['white and green', 'white and green', 'brown', 'white', 'white and blue', 'yellow', 'white and green', 'white and green', 'brown', None, None, None, None, None],
    'Size': [4, 1, 4, 2, 5, 3, 4, 5, 6, None, None, None, None, None],
    'Shape': ['cylinder', 'cylinder, short', 'cylinder', 'cylinder', 'cylinder, tall and slender', 'cylinder', 'cylinder', 'cuboid', 'cylinder, high', None, None, None, None, None],
    'Application': ['a milk product', 'a adhesive product', 'a coffee beverage', 'a container', 'a milk product', 'a refreshing beverage', 'a milk product', 'a refreshing beverage', 'a snack', None, None, None, None, None],
    'Other': ['a tapered mouth', None, None, None, None, 'a tapered mouth', 'green cap', None, 'yellow cap', None, None, None, None, None],
    'reshape': [(0.9,0.9,0.9),(1.2,1.2,1.2),(0.8,0.8,0.9),(0.8,0.8,1),(0.9,0.9,0.9),(1,1,1),(0.8,0.8,0.9),(0.7,0.7,1),(0.7,0.7,1), (0.6,0.6,0.6), (0.2,0.2,0.1), (0.9,0.9,1), (1,1,1), (1,1,1)],
    'mask_id':[1,2,6,7,9,8,10,11,12,14,13,15,16,17]
    }
    
    objs = pd.DataFrame(data)
    can_list = objs.ID.values
    can_list = can_list[can_list!=22]
    target_range={'Right':[[-50, -35],
                        [-20, 5],
                        [95, 105]],
                  'Left':[[-50, -35],
                        [-5, 20],
                        [95, 105]],
                }
    obj_range=[[-70, -30],
            [-30, 10],
            [95, 105]]
    joint_control = [-2,-3,-1,11,10,12,16,-4,-10,-9,-8,-7,9,3,4,5,6]
    def __init__(self,channel,scene_num = 1, map_id = 2):
        self.channel = channel
        self.scene_num = scene_num
        self.map_id = map_id
        self.sim_client=self.getSimFromClient(channel)
        self.initSim()
        time.sleep(1)
        self.setWorld(scene_num,map_id)
    
    def getSimFromClient(self,channel='127.0.0.1:30001'):
        channel = grpc.insecure_channel(channel,  # FIXME
                                    options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                                            ('grpc.max_receive_message_length', 1024 * 1024 * 1024)]) 
        sim_client = GrabSim_pb2_grpc.GrabSimStub(channel)
        return sim_client
    
    def initSim(self):
        self.sim_client.Init(GrabSim_pb2.NUL())

    def getMaps(self):
        message = self.sim_client.AcquireAvailableMaps(GrabSim_pb2.NUL())
        return message

    def setWorld(self,scene_num,map_id):
        initworld = self.sim_client.SetWorld(GrabSim_pb2.BatchMap(count=scene_num, mapID=map_id))
        time.sleep(0.5)
        return initworld
    
    def setLightIntensity(self,value=1.0):
        self.sim_client.SetLightIntensity(GrabSim_pb2.FloatValue(value=value))

    def getAvailableObjects(self):
        message = self.sim_client.AcquireTypes(GrabSim_pb2.NUL()).types
        return message    
    


class Sim(SimServer):
    def __init__(self,channel,scene_id):
        self.channel = channel
        self.sim_client=self.getSimFromClient(channel)
        self.scene_id = scene_id
        
        self.grasp_state={'Left':0,'Right':0}
        self.enableEndPointCtrl = True
        self.reset()

    def resetWorld(self):
        scene = self.sim_client.Reset(GrabSim_pb2.ResetParams(scene=self.scene_id))
        time.sleep(0.5)
        return scene

    def getObservation(self):
        scene = self.sim_client.Observe(GrabSim_pb2.SceneID(value=self.scene_id))
        return scene
    
    def getActuatorRanges(self):
        message = self.sim_client.GetActuatorRanges(GrabSim_pb2.SceneID(value=self.scene_id))
        actuators = []
        for actuator in message.actuators:
            actuators.append({'name':actuator.name,'lower':actuator.lower,'upper':actuator.upper})
        return actuators

    def getActuators(self):
        map_id=[0,1,2,3,6,9,12,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,
                    33,36,39,42,43,44,46,47,48]
        joints=self.getJoints(type='angle')
        values=[joints[id] for id in map_id]
        return np.array(values)

    def joint2actuator(self,joints):
        map_id=[0,1,2,3,6,9,12,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,
                    33,36,39,42,43,44,46,47,48]
        values=[joints[id] for id in map_id]
        return values
    
    def getJoints(self,type='angle'):
        assert type in ['full','angle','name']
        joints = self.getObservation().joints
        if type=='full':
            data = []
            for joint in joints:
                data.append({'name':joint.name,'angle':joint.angle})
            return data
        if type=='angle':
            joints = [joint.angle for joint in joints]
        else:
            joints = [joint.name for joint in joints]
        return joints

    def getState(self):
        state={}
        state['sensors']=self.getSensorsData('All',type='full')
        state['joints']=self.getJoints('full')
        return state

    def getCollision(self):
        scene = self.getObservation()
        return scene.collision

    def getSensorsData(self,handSide='Right',type='data'):
        assert handSide in ['All','Left','Right']
        assert type in ['data','full']
        message = self.sim_client.GetSensorDatas(GrabSim_pb2.SceneID(value=self.scene_id))
        if handSide=='All':
            sensors = message.sensors
        elif handSide=='Left':
            sensors = [message.sensors[2],message.sensors[0]]
        else:
            sensors = [message.sensors[3],message.sensors[1]]
        datas = []
        for sensor in sensors:
            if type=='data':
                datas.append(list(sensor.data))
            else:
                datas.append({'name':sensor.name,'data':list(sensor.data)})
        return datas

    def getObjsInfo(self):
        scene = self.getObservation()
        objLists = scene.objects
        objsInfo=[]
        for index,obj in enumerate(objLists):
            objInfo = {'name':obj.name}
            objInfo['location']=[obj.location.X,obj.location.Y,obj.location.Z]
            objInfo['rotation']=[obj.rotation.Roll,obj.rotation.Pitch,obj.rotation.Yaw]
            objInfo['ID'] = index
            objsInfo.append(objInfo)
        return objsInfo
    
    def addObjects(self, obj_list, location='relatively'):
        assert location in ['absolute','relatively']
        if location == 'absolute':
            X,Y = 0, 0
        else:
            scene = self.getObservation()
            X,Y = scene.location.X, scene.location.Y

        objs=[]
        for obj in obj_list:
            if len(obj)==4:
                obj += [0]*3
            if len(obj)==7:
                if (self.objs.ID==obj[0]).any():
                    obj += list(self.objs[self.objs.ID==obj[0]]['reshape'].values[0])
                else:
                    obj += [1]*3
            objs=[GrabSim_pb2.ObjectList.Object(type=obj[0], x=X + obj[1], y=Y + obj[2], z=obj[3], 
                                                roll=obj[4], pitch=obj[5], yaw=obj[6],
                                                sx=obj[7],sy=obj[8],sz=obj[9])]
            scene = self.sim_client.AddObjects(GrabSim_pb2.ObjectList(objects=objs, scene=self.scene_id))
            
            objLoc = self.getObjsInfo()[-1]['location']
            if self.desk_height is None:
                self.registry_objs.append(None)
            else:
                self.registry_objs.append((objLoc,objLoc[-1]-obj[3]))
            time.sleep(0.2)
        return scene

    def removeObjects(self,ids='all'):
        assert ids=='all' or isinstance(ids,list)
        if ids=='all':
            objs=self.getObjsInfo()
            ids=[i for i in range(len(objs))]
        ids.sort()
        for id in ids[::-1]:
            del self.registry_objs[id]
        scene=self.sim_client.RemoveObjects(GrabSim_pb2.RemoveList(IDs=ids, scene=self.scene_id))
        return scene

    def getImage(self,caremras=[GrabSim_pb2.CameraName.Head_Color,GrabSim_pb2.CameraName.Head_Segment]):
        action = GrabSim_pb2.CameraList(cameras=caremras)
        mats = []
        imgs = self.sim_client.Capture(action).images
        for im in imgs:
            mat = np.frombuffer(im.data,dtype=im.dtype).reshape((im.height, im.width, im.channels))
            mats.append(mat)
        return mats

    def EnableEndPointCtrl(self,enable=True):
        if enable != self.enableEndPointCtrl:
            if enable:
                hands=self.getSensorsData(handSide='Left')[0]
                self.moveHand(*hands,handSide='Left',gap=10,method='absolute')
                hands=self.getSensorsData(handSide='Right')[0]
                self.moveHand(*hands,handSide='Right',gap=10,method='absolute')
                time.sleep(0.05)
            else:
                values = self.getActuators()
                self.changeJoints(values,method='new')
                time.sleep(0.02)

        action = GrabSim_pb2.EnableEndPointCtrl(scene=self.scene_id,handSide=GrabSim_pb2.HandSide.Right,enable=enable)
        self.sim_client.SetEnableEndPointCtrl(action)
        action = GrabSim_pb2.EnableEndPointCtrl(scene=self.scene_id,handSide=GrabSim_pb2.HandSide.Left,enable=enable)
        self.sim_client.SetEnableEndPointCtrl(action)
        
        self.enableEndPointCtrl = enable

    def changeJoints(self,joints,method='new'):
        assert method in ['new','old']
        if method=='old':
            joints[14+2] *= -1
            map_id = [0,10,9,21,21,21,21,7,8,21,12,13,11,2,3,1,5,6,4,17,16,21,21,21,21,14,15,21,19,20,18]
            joints = [joints[id] for id in map_id]
        else:
            if len(joints)!=len(self.getActuatorRanges()):
                assert len(joints)==len(self.getJoints())
                joints = self.joint2actuator(joints)

        action = GrabSim_pb2.Action(scene=self.scene_id, action=GrabSim_pb2.Action.ActionType.RotateJoints, values=joints)
        message = self.sim_client.Do(action)

        time.sleep(0.01)
        return message

    def changeWrist(self,roll=0,pitch=0,yaw=0,handSide='Right'):
        assert handSide in ['Right','Left']
        
        values = self.getActuators()
        values[self.joint_control] = self.joints[self.joint_control]
        if handSide=='Right':
            hand_ids=[-2,-3,-1]
        else:
            hand_ids=[11,10,12]
        for id,angle in zip(hand_ids,[roll,pitch,yaw]):
            values[id]=angle

        self.changeJoints(values,method='new')
        values = self.getActuators()
        self.joints[hand_ids] = values[hand_ids]
    
    def getWrist(self,handSide='Right'):
        assert handSide in ['Right','Left']

        values = self.getActuators()
        if handSide=='Right':
            hand_ids=[-2,-3,-1]
        else:
            hand_ids=[11,10,12]
        pitch, roll, yaw = values[hand_ids[0]], values[hand_ids[1]], values[hand_ids[2]]
        return pitch, roll, yaw

    def getEndPointPosition(self,handSide='Right',x=0,y=0,z=0):
        '''
            pitch: X
            roll: Y
            yaw: Z
        '''
        assert handSide in ['Left','Right']
        if handSide == 'Left':
            action = GrabSim_pb2.EndPointPosition(scene=self.scene_id,handSide=GrabSim_pb2.HandSide.Left,x=x,y=y,z= z)
        else:
            action = GrabSim_pb2.EndPointPosition(scene=self.scene_id,handSide=GrabSim_pb2.HandSide.Right,x=x,y=y,z= z)
        return action

    def setEndPointPosition(self,action):
        self.sim_client.SetEndPointPosition(action)

    def moveHand(self,x=0,y=0,z=0,handSide='Right',method='diff',gap=0.3,keep_rpy=None):
        assert method in ['absolute','relatively','diff']
        ox,oy,oz = self.getSensorsData(handSide)[0]
        while ox==0 and oy==0 and oz==0:
            time.sleep(0.01)
            ox,oy,oz = self.getSensorsData(handSide)[0]
        if method == 'diff':
            x,y,z = x+ox, y+oy, z+oz
        
        elif method == 'relatively':
            scene = self.getObservation()
            X,Y = scene.location.X, scene.location.Y
            x += X
            y += Y
        if gap is None:
            action = self.getEndPointPosition(handSide,x,y,z)
            self.setEndPointPosition(action)
            time.sleep(0.05)
        else:        
            k = int(max(np.abs([x-ox,y-oy,z-oz]))/gap)+1
            lx,ly,lz=ox,oy,oz
            for index,(nx,ny,nz) in enumerate(np.linspace([ox,oy,oz],[x,y,z],k+1)[1:]):
                action = self.getEndPointPosition(handSide,nx,ny,nz)
                self.setEndPointPosition(action)
                time.sleep(0.03)
                if keep_rpy is not None and index%(int(3/gap) if int(3/gap)>0 else 1)==0:
                    self.set_world_rpy(keep_rpy,handSide=handSide)
                    time.sleep(0.05)
                lx,ly,lz=nx,ny,nz
        time.sleep(0.1)

    def moveHandReturnAction(self,x=0,y=0,z=0,handSide='Right',method='diff',gap=0.3,keep_rpy=None):
        assert method in ['absolute','relatively','diff']
        ox,oy,oz = self.getSensorsData(handSide)[0]
        while ox==0 and oy==0 and oz==0:
            time.sleep(0.01)
            ox,oy,oz = self.getSensorsData(handSide)[0]
        if method == 'diff':
            x,y,z = x+ox, y+oy, z+oz
        
        elif method == 'relatively':
            scene = self.getObservation()
            X,Y = scene.location.X, scene.location.Y
            x += X
            y += Y
        
        k = int(max(np.abs([x-ox,y-oy,z-oz]))/gap)+1
        lx,ly,lz=ox,oy,oz
        for index,(nx,ny,nz) in enumerate(np.linspace([ox,oy,oz],[x,y,z],k+1)[1:]):
            action = self.getEndPointPosition(handSide,nx,ny,nz)
            self.setEndPointPosition(action)
            time.sleep(0.1)
            if keep_rpy is not None and index%(int(3/gap) if int(3/gap)>0 else 1)==0:
                self.set_world_rpy(keep_rpy,handSide=handSide)
                time.sleep(0.05)
            if handSide=='Left':
                yield [nx-lx,ny-ly,nz-lz,0,0,0,self.grasp_state['Left'],self.grasp_state['Right']]
            else:
                yield [0,0,0,nx-lx,ny-ly,nz-lz,self.grasp_state['Left'],self.grasp_state['Right']]
            lx,ly,lz=nx,ny,nz
    
    def bow_head(self):
        values = self.getActuators()
        values[self.joint_control] = self.joints[self.joint_control]
        values[16]=35
        message=self.changeJoints(values,method='new')
        do_values = values
        values = self.getActuators()
        self.joints[16] = values[16]
        time.sleep(0.2)
        return do_values
    
    def grasp(self,type='grasp',angle=None,handSide='Right'):
        assert type in ['grasp','release',0,1]
        assert handSide in ['Right','Left']
        if type==0:
            type = 'release'
        if type==1:
            type = 'grasp'

        values = self.getActuators()
        values[self.joint_control] = self.joints[self.joint_control]
        if handSide=='Right':
            hand_ids=[-4,-10,-9,-8,-7]
        else:
            hand_ids=[9,3,4,5,6]
        if angle is None:
            if type=='grasp':
                angle=(65,68)
            else:
                angle=(-20,-20)

        if type == 'grasp':
            values[hand_ids[0]]=angle[0]
            values[hand_ids[1]]=angle[1]
            self.changeJoints(values,method='new')
            time.sleep(0.5)
            for id in hand_ids[2:]:
                values[id]=angle[1]
            self.changeJoints(values,method='new')
            time.sleep(0.2)
        else:
            for id in hand_ids[2:]:
                values[id]=angle[1]
            self.changeJoints(values,method='new')
            time.sleep(0.5)
            values[hand_ids[0]]=angle[0]
            values[hand_ids[1]]=angle[1]
            self.changeJoints(values,method='new')
            time.sleep(0.2)
            
        self.grasp_state[handSide]=1 if type=='grasp' else 0
        time.sleep(2)
        values = self.getActuators()
        self.joints[hand_ids] = values[hand_ids]

    def release(self,angle=None,handSide='Right'):
        self.grasp(type='release',angle=angle,handSide=handSide)

    def addDesk(self,desk_id=None,name=None,h=98,dx=0,dy=0):
        assert (desk_id is not None) or (name is not None)
        if desk_id is not None:
            loc=self.desks[self.desks['ID']==desk_id].values[0][-3:] 
        elif name is not None:
            loc=self.desks[self.desks['Name']==name].values[0][-3:] 
            desk_id=self.desks[self.desks['Name']==name].values[0][0]
        desk = [desk_id,loc[1],loc[2],h-loc[0],0,0,0]
        desk[1]+=dx
        desk[2]+=dy
        objList = [desk]
        self.addObjects(objList,location='relatively')
        self.desk_height = h

    def clearObjs(self):
        objs=self.getObjsInfo()
        ids=[]
        for i,obj in enumerate(objs):
            if obj['name'] in self.objs.Name.values:
                ids.append(i)
        self.removeObjects(ids=ids)
     
    def genObjs(self,n=1,handSide='Right',ids=None,same_objs=False,target_loc=None,forbid_obj_ids=None,h=90,min_distance=15,retry_times=20):
        if ids is None:
            if forbid_obj_ids is not None:
                values = self.objs.ID[~self.objs.ID.isin(forbid_obj_ids)].values
            else:
                values = self.objs.ID.values
            if same_objs==False:
                ids = random.sample(list(values),n)
            else:
                ids = random.choices(list(values),k=n)
        elif isinstance(ids,int):
            ids = [ids]*n
        objs=[]
        objs_loc = generate_points_in_square(n,self.target_range[handSide],self.obj_range,target_loc=target_loc,min_distance=min_distance,retry_times=retry_times)
        for id,loc in zip(ids,objs_loc):
            objs.append([id,loc[0],loc[1],h+1,0,0,0])
        self.addObjects(objs,location='relatively')
        self.gen_objs = objs
        return objs
    
    def findObj(self,name=None,id=None):
        assert (name is not None) or (id is not None)

        objInfo = self.getObjsInfo()
        if id is not None:
            return objInfo[id]
        else:
            for index,obj in enumerate(objInfo):
                if obj['name']==name:
                    return obj
    
    def closeTargetObj(self,obj_id,handSide='Right',gap=1,keep_rpy=(0,0,0)):
        if handSide=='Right':
            obj_loc=np.array(self.findObj(id=obj_id)['location'])
            obj_loc[0]+=4
            obj_loc[1]-=6
            obj_loc[2] = self.desk_height+5.5
            for i in range(100):
                sensor = self.getSensorsData(handSide='All',type='full')
                middle = np.array(sensor[-10]['data'])
                p = max(abs(obj_loc-middle))/gap if max(abs(obj_loc-middle))>gap else 1
                vector = (obj_loc-middle)/p
                if max(abs(obj_loc[:2]-middle[:2]))<1 and max(abs(obj_loc[2:]-middle[2:]))<2:
                    break
                self.moveHand(*vector,handSide=handSide,method='diff',gap=gap,keep_rpy=(0,0,0))
                yield [0,0,0,*vector,self.grasp_state['Left'],self.grasp_state['Right']]
            obj_loc=np.array(self.findObj(id=obj_id)['location'])
            obj_loc[0]-=4+2
            obj_loc[1]-=2-1
            obj_loc[2] = self.desk_height+5.5
            for i in range(20):
                sensor = self.getSensorsData(handSide='All',type='full')
                middle = np.array(sensor[-10]['data'])
                p = max(abs(obj_loc-middle))/gap if max(abs(obj_loc-middle))>gap else 1
                vector = (obj_loc-middle)/p
                if max(abs(obj_loc[:2]-middle[:2]))<1 and max(abs(obj_loc[2:]-middle[2:]))<2:
                    break
                self.moveHand(*vector,handSide=handSide,method='diff',gap=0.1,keep_rpy=(0,0,0))
                yield [0,0,0,*vector,self.grasp_state['Left'],self.grasp_state['Right']]
        elif handSide=='Left':
            obj_loc=np.array(self.findObj(id=obj_id)['location'])
            obj_loc[0]+=4
            obj_loc[1]+=6
            obj_loc[2] = self.desk_height+5.5
            for i in range(100):
                sensor = self.getSensorsData(handSide='All',type='full')
                middle = np.array(sensor[-24]['data'])
                p = max(abs(obj_loc-middle))/gap if max(abs(obj_loc-middle))>gap else 1
                vector = (obj_loc-middle)/p
                if max(abs(obj_loc[:2]-middle[:2]))<1 and max(abs(obj_loc[2:]-middle[2:]))<2:
                    break
                self.moveHand(*vector,handSide=handSide,method='diff',gap=gap,keep_rpy=(0,0,0))
                yield [*vector,0,0,0,self.grasp_state['Left'],self.grasp_state['Right']]
            obj_loc=np.array(self.findObj(id=obj_id)['location'])
            obj_loc[0]-=4
            obj_loc[1]+=4
            obj_loc[2] = self.desk_height+5.5
            for i in range(20):
                sensor = self.getSensorsData(handSide='All',type='full')
                middle = np.array(sensor[-24]['data'])
                p = max(abs(obj_loc-middle))/gap if max(abs(obj_loc-middle))>gap else 1
                vector = (obj_loc-middle)/p
                if max(abs(obj_loc[:2]-middle[:2]))<1 and max(abs(obj_loc[2:]-middle[2:]))<2:
                    break
                self.moveHand(*vector,handSide=handSide,method='diff',gap=0.1,keep_rpy=(0,0,0))
                yield [*vector,0,0,0,self.grasp_state['Left'],self.grasp_state['Right']]

    def set_world_rpy(self,world_rpy_value,handSide='Right'):
        world_rpy = euler_from_quaternion(self.getSensorsData(handSide=handSide)[1])
        robot_rpy = self.getWrist(handSide=handSide)
        transformation_matrix = get_transformation_matrix(world_rpy, robot_rpy)
        r,p,y = world_rpy_to_robot_rpy(world_rpy_value,transformation_matrix,handSide=handSide)
        self.changeWrist(roll=r,pitch=p,yaw=y,handSide=handSide)
        return r,p,y

    def initState(self):
        self.EnableEndPointCtrl(False)
        joints=[1.8061175069306046e-05,-98.61856842041016,4.662295341491699,-1.788636326789856,-1.789079189300537,-1.7891252040863037,
                        -1.7886359691619873,-1.7890788316726685,-1.7891250848770142,-1.7886145114898682,-1.789078712463379,-1.7891266345977783,
                        -1.7886520624160767,-1.78908371925354,-1.7891193628311157,-52.2512092590332,21.009044647216797,-2.2276458740234375,
                        -2.230302333831787,30.00031089782715,6.811821460723877,22.398529052734375,-0.8751718401908875,0.14142996072769165,
                        -0.007084177806973457,35.264503479003906,-0.0020231136586517096,5.13994054927025e-05,-100.19393157958984,-4.168997287750244,
                        -2.1731982231140137,-2.1735899448394775,-2.173584222793579,-2.1731984615325928,-2.1735899448394775,-2.173584461212158,
                        -2.173130512237549,-2.1735763549804688,-2.173611640930176,-2.1731984615325928,-2.1735899448394775,-2.173584461212158,
                        -51.53587341308594,-21.0319881439209,-1.1483358144760132,-1.1480298042297363,30.00035858154297,-5.126368522644043,-25.800668716430664]
        self.joints = np.array(self.joint2actuator(joints))
        self.changeJoints(joints)
        time.sleep(0.5)
        self.EnableEndPointCtrl(True)

        self.grasp_state={'Left':0,'Right':0}
        self.release(handSide='Left')
        self.release(handSide='Right')

    def reset(self,deskID=None,h=None,n_objs=1,obj_id=None):
        self.resetWorld()
        self.registry_objs = [None]*len(self.getObjsInfo())
        self.removeObjects(ids='all')
        time.sleep(0.5)
        sensor_data=self.getSensorsData('All')

        self.initState()
        self.bow_head()
        time.sleep(1)

        self.desk_height = None

class SimAction(Sim):
    def __init__(self,channel,scene_id):
        super().__init__(channel,scene_id)
    
    def graspTargetObj(self,obj_id,handSide='Right',gap=1,keep_rpy=(0,0,0),distance=10):
        self.release(handSide=handSide)
        for action in self.closeTargetObj(obj_id=obj_id,handSide=handSide,gap=gap,keep_rpy=keep_rpy):
            yield (action,'close to')
        self.grasp(handSide=handSide)
        if handSide=='Left':
            action = [0]*6+[1,0]
        else:
            action = [0]*6+[0,1]
        yield (action, 'grasp')
        height = distance
        for action in self.moveHandReturnAction(0,0,height,handSide=handSide,gap=gap):
            yield (action, 'move up')

    def checkGraspTargetObj(self,obj_id,height=5):
        target_loc=self.getObjsInfo()[obj_id]['location']
        desk_height = self.desk_height
        if desk_height+self.registry_objs[obj_id][1]+height<target_loc[-1]:
            return True
        else:
            return False
    
    def placeTargetObj(self,obj_id,handSide='Right',gap=1,keep_rpy=(0,0,0)):
        if not self.checkGraspTargetObj(obj_id):
            return
        if handSide=='Right':
            obj_loc=np.array(self.findObj(id=obj_id)['location'])
            obj_loc[2] = self.desk_height+self.registry_objs[obj_id][1]+3
            for i in range(100):
                sensor = self.getSensorsData(handSide='All',type='full')
                middle = np.array(sensor[-10]['data'])
                p = max(abs(obj_loc-middle))/gap if max(abs(obj_loc-middle))>gap else 1
                vector = (obj_loc-middle)/p
                if max(abs(obj_loc[:2]-middle[:2]))<1 and max(abs(obj_loc[2:]-middle[2:]))<2:
                    break
                self.moveHand(*vector,handSide=handSide,method='diff',gap=gap,keep_rpy=(0,0,0))
                yield ([0,0,0,*vector,self.grasp_state['Left'],self.grasp_state['Right']], 'move down')
        elif handSide=='Left':
            obj_loc=np.array(self.findObj(id=obj_id)['location'])
            obj_loc[2] = self.desk_height+self.registry_objs[obj_id][1]+3
            for i in range(100):
                sensor = self.getSensorsData(handSide='All',type='full')
                middle = np.array(sensor[-24]['data'])
                p = max(abs(obj_loc-middle))/gap if max(abs(obj_loc-middle))>gap else 1
                vector = (obj_loc-middle)/p
                if max(abs(obj_loc[:2]-middle[:2]))<1 and max(abs(obj_loc[2:]-middle[2:]))<2:
                    break
                self.moveHand(*vector,handSide=handSide,method='diff',gap=gap,keep_rpy=(0,0,0))
                yield ([*vector,0,0,0,self.grasp_state['Left'],self.grasp_state['Right']], 'move down')
        self.release(handSide=handSide)
        yield ([0]*6+[self.grasp_state['Left'],self.grasp_state['Right']], 'release')

    def checkPlaceTargetObj(self,obj_id):
        target_loc=self.getObjsInfo()[obj_id]['location']
        desk_height = self.desk_height
        if target_loc[-1]-(desk_height+self.registry_objs[obj_id][1])<=3:
            return True
        else:
            return False
        
    def moveNear(self,obj1_id,obj2_id,distance=10,handSide='Right',gap=1,keep_rpy=(0,0,0)):
        for action in self.graspTargetObj(obj_id=obj1_id,handSide=handSide,gap=gap,keep_rpy=keep_rpy):
            yield action
        time.sleep(0.2)
        if not self.checkGraspTargetObj(obj_id=obj1_id):
            return
        obj1_loc = np.array(self.getObjsInfo()[obj1_id]['location'])
        obj2_loc = np.array(self.getObjsInfo()[obj2_id]['location'])

        vector = obj2_loc - obj1_loc
        unit_vector = vector / np.linalg.norm(vector)
        new_position = obj2_loc - unit_vector * distance
        gap=0.5
        for action in self.moveHandReturnAction(*(new_position-obj1_loc),handSide=handSide,gap=gap):
            yield (action, 'close to')
    
    def checkMoveNear(self,obj1_id,obj2_id,distance=11):
        obj1_loc = np.array(self.getObjsInfo()[obj1_id]['location'])[:2]
        obj2_loc = np.array(self.getObjsInfo()[obj2_id]['location'])[:2]
        dis = np.linalg.norm(obj2_loc-obj1_loc)
        if dis<distance:
            return True
        return False
    
    def knockOver(self,obj_id,handSide='Right',gap=1,keep_rpy=(0,0,0)):
        for action in self.closeTargetObj(obj_id,handSide=handSide,gap=gap,keep_rpy=keep_rpy):
            yield (action, 'close to')
        if handSide=='Left':
            diff = (0,-10,10)
        else:
            diff = (0,10,10)
        for action in self.moveHandReturnAction(*diff,handSide=handSide,gap=2,keep_rpy=keep_rpy):
            yield (action, 'push front')

    def checkKnockOver(self,obj_id):
        obj_loc=np.array(self.findObj(id=obj_id)['location'])
        if self.desk_height+self.registry_objs[obj_id][1]>obj_loc[-1]+2:
            return True
        return False
    
    def pushFront(self,obj_id,distance=5,handSide='Right',gap=1,keep_rpy=(0,0,0)):
        for action in self.closeTargetObj(obj_id,handSide=handSide,gap=gap,keep_rpy=keep_rpy):
            yield (action, 'close to')
        gap=0.2
        for action in self.moveHandReturnAction(-distance,0,0,handSide=handSide,gap=gap,keep_rpy=keep_rpy):
            yield (action, 'move front')
    
    def checkPushFront(self,obj_id,distance=3):
        initLoc = self.registry_objs[obj_id][0]
        nowLoc = self.getObjsInfo()[obj_id]['location']
        if nowLoc[0]-initLoc[0]<-distance:
            return True
        return False
    
    def pushLeft(self,obj_id,distance=5,handSide='Right',gap=1,keep_rpy=(0,0,0)):
        assert handSide=='Right'
        for action in self.closeTargetObj(obj_id,handSide=handSide,gap=gap,keep_rpy=keep_rpy):
            yield (action, 'close to')
        gap=0.2
        for action in self.moveHandReturnAction(0,distance,0,handSide=handSide,gap=gap,keep_rpy=keep_rpy):
            yield (action, 'move left')

    def checkPushLeft(self,obj_id,distance=3):
        initLoc = self.registry_objs[obj_id][0]
        nowLoc = self.getObjsInfo()[obj_id]['location']
        if nowLoc[1]-initLoc[1]>distance:
            return True
        return False
    
    def pushRight(self,obj_id,distance=5,handSide='Left',gap=1,keep_rpy=(0,0,0)):
        assert handSide=='Left'
        for action in self.closeTargetObj(obj_id,handSide=handSide,gap=gap,keep_rpy=keep_rpy):
            yield (action, 'close to')
        gap=0.2
        for action in self.moveHandReturnAction(0,-distance,0,handSide=handSide,gap=gap,keep_rpy=keep_rpy):
            yield (action, 'move right')

    def checkPushRight(self,obj_id,distance=3):
        initLoc = self.registry_objs[obj_id][0]
        nowLoc = self.getObjsInfo()[obj_id]['location']
        if nowLoc[1]-initLoc[1]<-distance:
            return True
        return False
    
def generate_points_in_square(num_points,target_range, obj_range, target_loc=None, min_distance=15,retry_times=20):
    points = []
    range_x = obj_range[0]
    range_y = obj_range[1]
    for tried_times in range(retry_times):
        points = []
        if target_loc is None:
            x = np.random.uniform(target_range[0][0], target_range[0][1])
            y = np.random.uniform(target_range[1][0], target_range[1][1])
            new_point = np.array([x, y])
            points.append(new_point)
        else:
            new_point = np.array(target_loc)
            points.append(new_point)
        range_x,range_y = target_range[:2]
        if len(points)==num_points:
            return np.array(points)
        for _ in range(num_points):
            find_times=0

            while True:
                x = np.random.uniform(range_x[0], range_x[1])
                y = np.random.uniform(range_y[0], range_y[1])
                new_point = np.array([x, y])
                
                if len(points)>0:
                    distances = np.linalg.norm(new_point - np.array(points), axis=1)
                else:
                    distances = []
                if len(distances) == 0 or all(distances > min_distance):
                    points.append(new_point)
                    break
                find_times+=1
                if find_times>=retry_times:
                    range_x,range_y = obj_range[:2]
                    break
            if len(points)==num_points:
                return np.array(points)
    print(f'Fail generate {num_points}, total generated objs number is {len(points)}')
    return np.array(points)

def euler_from_quaternion(quaternion):
    x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    roll_z = math.atan2(t3, t4)

    angle = 180/np.pi
    return roll_x*angle, pitch_y*angle, roll_z*angle
 
def rotation_matrix_to_rpy(rotation_matrix):
    rpy = tf.euler.mat2euler(rotation_matrix, 'sxyz')
    return rpy

def rpy_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])

    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])

    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def transform_robot(robot_rpy,handSide='Right'):
    robot_rpy*=-1
    robot_rpy[0],robot_rpy[2]=robot_rpy[2],robot_rpy[0]
    return robot_rpy

def world_rpy_to_robot_rpy(world_rpy,transformation_matrix,handSide='Right'):
    if not isinstance(world_rpy,np.ndarray):
        world_rpy=np.array(world_rpy)
    world_rpy = world_rpy/180*np.pi
    x=transformation_matrix.dot(rpy_to_rotation_matrix(world_rpy[0],world_rpy[1],world_rpy[2]))
    x=rotation_matrix_to_rpy(x)
    x=np.array(x)/np.pi*180
    x=transform_robot(x,handSide=handSide)
    return x

def get_transformation_matrix(world_rpy, robot_rpy):
    if not isinstance(world_rpy,np.ndarray):
        world_rpy=np.array(world_rpy)
    if not isinstance(robot_rpy,np.ndarray):
        robot_rpy=np.array(robot_rpy)
    world_rpy = world_rpy/180*np.pi
    robot_rpy = robot_rpy/180*np.pi   
    robot_rpy = transform_robot(robot_rpy)
    world_roll, world_pitch, world_yaw = world_rpy
    robot_roll, robot_pitch, robot_yaw = robot_rpy

    world_to_robot = np.dot(rpy_to_rotation_matrix(robot_roll, robot_pitch, robot_yaw),
                            np.linalg.inv(rpy_to_rotation_matrix(world_roll, world_pitch, world_yaw)))

    return world_to_robot
