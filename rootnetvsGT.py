'''
some parts are copied from 
https://github.com/facebookresearch/InterHand2.6M/blob/67ba1b8e2c8da0f79ba8a3de5bb401714b4ebea2/data/InterHand2.6M/dataset.py
'''

import json
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

interhand_joint_path=r"C:\Users\lahir\Downloads\InterHand2.6M_5fps_batch1\annotations\val\val\InterHand2.6M_val_joint_3d.json"
interhand_data_path=r"C:\Users\lahir\Downloads\InterHand2.6M_5fps_batch1\annotations\val\val\InterHand2.6M_val_data.json"
interhand_camera_path=r"C:\Users\lahir\Downloads\InterHand2.6M_5fps_batch1\annotations\val\val\InterHand2.6M_val_camera.json"
rootnet_path=r"C:\Users\lahir\Downloads\rootnet_interhand2.6m_output_val_30fps.json"
root_joint_idx = {'right': 20, 'left': 41}

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

#read RootNet data
rootnet_result = {}
with open(rootnet_path) as f:
    annot = json.load(f)
for i in range(len(annot)):
    rootnet_result[str(annot[i]['annot_id'])] = annot[i]

with open(interhand_joint_path) as f:
    joints = json.load(f)

with open(interhand_camera_path) as f:
        cameras = json.load(f)

db = COCO(interhand_data_path)


right_RN_list,right_GT_list,left_RN_list,left_GT_list=[],[],[],[]
error=[]
for aid in db.anns.keys():
    #read the RootNet output
    abs_depth_RN = {'right': rootnet_result[str(aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
    right_RN_list.append(abs_depth_RN['right'] )
    left_RN_list.append(abs_depth_RN['left'] )

    #read InterHands output
    ann = db.anns[aid]
    image_id = ann['image_id']
    img = db.loadImgs(image_id)[0]
    capture_id = img['capture']
    frame_idx = img['frame_idx']
    cam = img['camera']
    joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
    campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
    joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
    abs_depth_GT = {'right': joint_cam[root_joint_idx['right'],2], 'left': joint_cam[root_joint_idx['left'],2]}
    
    right_GT_list.append(abs_depth_GT['right'])
    left_GT_list.append(abs_depth_GT['left'])

    error.append(abs(abs_depth_RN['right']-abs_depth_GT['right']))


vals=np.column_stack((right_RN_list,right_GT_list))
sortargs=vals[:,1].argsort()

plt.plot(vals[sortargs,0])
plt.plot(vals[sortargs,1])
plt.show()

np.mean(error)


