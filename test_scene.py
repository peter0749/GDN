import os
import sys
import json
import warnings
from tensorboardX import SummaryWriter
from tqdm import tqdm

import numba as nb
warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#torch.backends.cudnn.benchmark = True

from gdn.utils import *
from gdn.detector.utils import *
from gdn import import_model_by_setting
import importlib
import copy
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to configuration file (in JSON)")
    parser.add_argument("weights", type=str, help="Path to weights")
    parser.add_argument("input_points", type=int, help="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    #mp.set_start_method('spawn', force=True)
    mp.set_start_method('forkserver', force=True)
    from mayavi import mlab
    from nms import decode_euler_feature
    from nms import initEigen, sanity_check
    from nms import crop_index, generate_gripper_edge
    from scipy.spatial.transform import Rotation
    import time

    args = parse_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)
        deepen_hand = config['hand_height']
        config['input_points'] = args.input_points

    #representation, dataset, my_collate_fn, model, _, __, ___ = import_model_by_setting(config, mode='train')
    representation, dataset, my_collate_fn, model, _, __, ___ = import_model_by_setting(config, mode='eval')
    model.load_state_dict(torch.load(args.weights)['base_model'])

    #dataset.eval() # with mode 'train'
    model.eval()
    pc_npy, poses = dataset[0]

    feat_ts = time.time()
    start_ts = feat_ts
    with torch.no_grad():
        pc = torch.from_numpy(pc_npy).float()
        pred = model(pc.unsqueeze(0).cuda())[0].cpu().numpy().astype(np.float32)

    pred_poses = np.asarray(decode_euler_feature(
          pc_npy,
          pred.reshape(1,-1),
          *pred.shape[:-1],
          config['hand_height'],
          config['gripper_width'],
          config['thickness_side'],
          30.0,
          0.02,
          5000, # max number of candidate
          -np.inf, # threshold of candidate
          1000,  # max number of grasp in NMS
          5,    # number of threads
          True  # use NMS
        ), dtype=np.float32)
    filter_ts = time.time()
    print("Decode in %.2f seconds."%(filter_ts-feat_ts))
    pred_poses = sanity_check(pc_npy, pred_poses, 0,
            config['gripper_width'],
            config['thickness'],
            config['hand_height'],
            config['thickness_side'],
            5 # num threads
            )
    end_ts = time.time()
    print("Filter in %.2f seconds."%(end_ts-filter_ts))
    print('Generated1 %d grasps in %.2f seconds.'%(len(pred_poses), end_ts-start_ts))

    fig_pred = mlab.figure()
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], scale_factor=0.004, mode='sphere', color=(0.0,1.0,1.0), opacity=1.0, figure=fig_pred)
    for n, pose in enumerate(pred_poses[:100]):
        gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'],
                                                                                   config['hand_height'],
                                                                                   pose,
                                                                                   config['thickness_side'],
                                                                                   0.0
                                                                                   )
        gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge

        mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=config['thickness']/4., color=(0,0,1), opacity=0.5, figure=fig_pred)
        mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=config['thickness']/4., color=(0,0,1), opacity=0.5, figure=fig_pred)
        mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=config['thickness']/4., color=(0,0,1), opacity=0.5, figure=fig_pred)

    fig_gt = mlab.figure()
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], scale_factor=0.004, mode='sphere', color=(0.0,1.0,1.0), opacity=1.0, figure=fig_gt)
    for n, pose in enumerate(poses[:100]):
        gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'],
                                                                                   config['hand_height'],
                                                                                   pose.astype(np.float32),
                                                                                   config['thickness_side'],
                                                                                   0.0
                                                                                   )
        gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge

        mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=config['thickness']/4., color=(0,1,0), opacity=0.5, figure=fig_gt)
        mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=config['thickness']/4., color=(0,1,0), opacity=0.5, figure=fig_gt)
        mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=config['thickness']/4., color=(0,1,0), opacity=0.5, figure=fig_gt)

    mlab.show()
