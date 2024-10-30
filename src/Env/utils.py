import numpy as np
import pickle
from collections import OrderedDict
from . import GrabSim_pb2

def initJointsArrange():
    joints_arrange = [
        [-36,30],
        [-90,90],
        [-45,45], 
        [-45,45], 
        [-180,180], 
        [-45,36], 
        [-23,23],

        [-180,36], 
        [-23,90], 
        [-90,90], 
        [-120,12], 
        [-90,90], 
        [-23,23],
        [-36,23], 

        [-180,36],
        [-90,23], 
        [-90,90], 
        [-120,12],
        [-90,90],
        [-23,23],
        [-23,36],
    ]
    return np.array(joints_arrange,dtype=np.float32)

def get_instructions():
    f=open('instructions/database.pkl','rb')
    instructions=pickle.load(f)
    return instructions

def check_mask_id(sim):
    sim.addDesk(1)
    for id in sim.objs.ID.values:
        scene=sim.removeObjects([i for i in range(1,len(sim.registry_objs))])
        sim.genObjs(n=1,ids=int(id),h=sim.desk_height,handSide='Right')
        import matplotlib.pyplot as plt
        import numpy as np
        caremras=[GrabSim_pb2.CameraName.Head_Segment]
        ignore = np.array([0,128])
        mat=sim.getImage(caremras)
        assert (mat==ignore[0]).any() and (mat==ignore[1]).any(), 'mask may have been changed'
        unique_values = np.unique( mat.ravel())
        unique_values = np.setdiff1d(unique_values,ignore)
        assert len(unique_values)==1, 'there are other objs'
        print(id,unique_values[0])