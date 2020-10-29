# coding: utf-8

import os
import sys
import json
import pickle
import warnings
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

import numpy as np
import numba as nb
warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True

from pointnet2.utils import pointnet2_utils

from gdn.utils import *
from gdn.detector.utils import *
from gdn import import_model_by_setting

from nms import decode_euler_feature
from nms import initEigen, sanity_check
initEigen(0)

import importlib
import copy
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to configuration file (in JSON)")
    parser.add_argument("weight_path", type=str, help="Path to the trained model weights")
    parser.add_argument("pc_path", type=str, help="Path to the point cloud files")
    parser.add_argument("n_iteration", type=int, help="")
    parser.add_argument("n_proposal", type=int, help="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    #mp.set_start_method('forkserver', force=True)

    args = parse_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)
    config['point_cloud_path'] = args.pc_path

    representation, dataset, my_collate_fn, base_model, model, _, __ = import_model_by_setting(config, mode='val')
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=True,
                            shuffle=True,
                            collate_fn=my_collate_fn)
    print('Num params: %d'%count_parameters(base_model))

    states = torch.load(args.weight_path)
    base_model.load_state_dict(states['base_model'])

    model.eval()
    batch_iterator = iter(dataloader)
    inferenced_pcs = 0
    model_inference_time_ave = 0
    nms_time_ave = 0
    single_object_inference_time_ave = 0
    start_ts = time.time()
    with torch.no_grad():
        while True:
            if inferenced_pcs>=args.n_iteration:
                break
            start_single_object = time.time()
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break

            pc, gt, pc_origin, scene_ids = batch
            pc_cuda = pc.cuda()
            pred, ind, att = model(pc_cuda)
            model_inference_ts = time.time()

            pred = pred.cpu().numpy()
            pc_subsampled = pointnet2_utils.gather_operation(pc_cuda.transpose(1, 2).contiguous(), ind)
            pc_npy = pc_subsampled.cpu().transpose(1, 2).numpy() + pc_origin # (B, N, 3) + (B, 1, 3)
            pred_poses = np.asarray(decode_euler_feature(
                    pc_npy[0],
                    pred[0].reshape(1,-1),
                    *pred[0].shape[:-1],
                    config['hand_height'],
                    config['gripper_width'],
                    config['thickness_side'],
                    config['rot_th'],
                    config['trans_th'],
                    1000, # max number of candidate
                    -np.inf, # threshold of candidate
                    args.n_proposal,  # max number of grasp in NMS
                    config['num_workers'],    # number of threads
                    True  # use NMS
                  ), dtype=np.float32)
            pred_poses = sanity_check(pc_npy[0], pred_poses, 2,
                      config['gripper_width'],
                      config['thickness'],
                      config['hand_height'],
                      config['thickness_side'],
                      config['num_workers'] # num threads
                      )
            '''
            pred_poses = representation.retrive_from_feature_volume_batch(pc_npy, pred, n_output=args.n_proposal, threshold=-1e4, nms=True)
            pred_poses = representation.filter_out_invalid_grasp_batch(pc_npy, pred_poses)
            '''
            model_nms_ts = time.time()

            model_inference_time = model_inference_ts - start_single_object
            nms_time = model_nms_ts - model_inference_ts
            single_object_inference_time = model_nms_ts - start_single_object
            model_inference_time_ave += model_inference_time
            nms_time_ave += nms_time
            single_object_inference_time_ave += single_object_inference_time
            inferenced_pcs += len(pc)
    end_ts = time.time()
    elapsed_time = end_ts - start_ts
    ave_time = elapsed_time / float(inferenced_pcs)
    nms_time_ave = nms_time_ave / float(inferenced_pcs)
    model_inference_time_ave = model_inference_time_ave / float(inferenced_pcs)
    single_object_inference_time_ave = single_object_inference_time_ave / float(inferenced_pcs)
    print("model\tnms\tinf\teps\tave")
    print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f"%(model_inference_time_ave, nms_time_ave, single_object_inference_time_ave, elapsed_time, ave_time))
