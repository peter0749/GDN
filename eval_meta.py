# coding: utf-8

import os
import sys
import json
import pickle
import warnings
from tensorboardX import SummaryWriter
from tqdm import tqdm

import numpy as np
import numba as nb
warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True

from pointnet2.utils import pointnet2_utils

from nms import decode_euler_feature
from nms import initEigen, sanity_check
initEigen(0)

from gdn.utils import *
from gdn.detector.utils import *
from gdn import import_model_by_setting
import importlib
import copy
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to configuration file (in JSON)")
    parser.add_argument("weight_path", type=str, help="Path to the trained model weights")
    parser.add_argument("pc_path", type=str, help="Path to the point cloud files")
    parser.add_argument("shot", type=int, help="Number of shots")
    parser.add_argument("output_dir", type=str, help="Path to the results")
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    representation, dataset, my_collate_fn, base_model, model, _, __ = import_model_by_setting(config, mode='val')
    if config['representation'] == 'euler_scene_att_ce_meta_cd':
        dataset_query, _, dataset_support = dataset
    else:
        dataset_query, dataset_support = dataset
    dataset_query.eval()
    dataset_support.train() # use train set as support set
    dataloader_query = DataLoader(dataset_query,
                            batch_size=config['query_batch_size'],
                            num_workers=2,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=my_collate_fn)
    dataloader_support = DataLoader(dataset_support,
                            batch_size=args.shot,
                            num_workers=2,
                            pin_memory=True,
                            shuffle=True,
                            collate_fn=my_collate_fn)
    print('Num params: %d'%count_parameters(base_model))

    states = torch.load(args.weight_path)
    base_model.load_state_dict(states['base_model'])

    model.eval()
    batch_iterator = iter(dataloader_query)
    support_iterator = iter(dataloader_support)
    with torch.no_grad():
        while True:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break
            try:
                support_batch = next(support_iterator)
            except StopIteration:
                support_iterator = iter(dataloader_support)
                support_batch = next(support_iterator)
            pc_query, _, _, _, pc_origin_query, scene_ids = batch
            pc_support, _, mask_support, gt_support, _, support_id = support_batch

            pc_query = pc_query.cuda()
            pc_support = pc_support.cuda()
            mask_support = mask_support.cuda()

            pred, ind, att = model(pc_support, mask_support, pc_query, None)[:3]
            pc_subsampled = pointnet2_utils.gather_operation(pc_query.transpose(1, 2).contiguous(), ind)
            pc_npy = pc_subsampled.cpu().transpose(1, 2).numpy() + pc_origin_query # (B, N, 3) + (B, 1, 3)
            pc_npy = pc_npy.astype(np.float32)
            pred = pred.cpu().numpy().astype(np.float32)
            pred_poses = [
                    np.asarray(sanity_check(pc_npy[b],
                        np.asarray(decode_euler_feature(
                        pc_npy[b],
                        pred[b].reshape(1,-1),
                        *pred[b].shape[:-1],
                        config['hand_height'],
                        config['gripper_width'],
                        config['thickness_side'],
                        config['rot_th'],
                        config['trans_th'],
                        3000, # max number of candidate
                        -np.inf, # threshold of candidate
                        3000,  # max number of grasp in NMS
                        config['num_workers'],    # number of threads
                        True  # use NMS
                        ), dtype=np.float32)
                        , 2,
                        config['gripper_width_r'],
                        config['thickness_r'],
                        config['hand_height_r'],
                        config['thickness_side_r'],
                        config['num_workers'] # num threads
                    ), dtype=np.float32) for b in range(len(pc_npy)) ]
            #pred_poses = representation.retrive_from_feature_volume_batch(pc_npy, pred.cpu().numpy(), n_output=300, threshold=-1e8, nms=True)
            #pred_poses = representation.filter_out_invalid_grasp_batch(pc_npy, pred_poses)
            for pose, id_ in zip(pred_poses, scene_ids):
                prefix = args.output_dir
                meta = [(-n, id_, n) for n in range(len(pose))]
                with open(prefix+'/'+id_+'.meta', 'wb') as fp:
                    pickle.dump(meta, fp, protocol=2, fix_imports=True)
                np.save(prefix+'/'+id_+'.npy', pose)
                with open(prefix+'/'+id_+'.support', 'wb') as fp:
                    pickle.dump({'scene_ids': support_id, 'poses': np.array(gt_support, dtype=np.float32)}, fp, protocol=2, fix_imports=True)
                print('Processed: {} {:s}'.format(id_, str(pose.shape)))