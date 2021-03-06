# coding: utf-8

from mayavi import mlab
mlab.options.offscreen = True

import os
import sys
import json
import pickle
import warnings
from tqdm import tqdm

import numpy as np
import numba as nb
warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True

from sklearn.cluster import KMeans
from dppy.finite_dpps import FiniteDPP

from nms import decode_euler_feature
from nms import initEigen, sanity_check
initEigen(0)

from gdn.utils import *
from gdn.detector.utils import *
from gdn import import_model_by_setting
import importlib
import copy
from argparse import ArgumentParser

@nb.njit
def rotDist(x, y):
    return np.arcsin(np.linalg.norm(np.eye(3) - np.dot(x[:3,:3], y[:3,:3].T), ord=None ) / (2*np.sqrt(2)))

@nb.njit
def makeAffinityMatrix(poses, sigma_r=0.5, sigma_t=0.25):
    A = np.eye(len(poses), dtype=np.float32)
    for i in range(len(poses)):
        for j in range(i+1, len(poses)):
            d = rotDist(poses[i,:3,:3], poses[j,:3,:3])
            s_r = np.exp(-d / ((2. * sigma_r) ** 2))
            if not np.isfinite(s_r):
                s_r = 0.0
            d = np.linalg.norm(poses[i,:3,3]-poses[j,:3,3], ord=2)
            s_t = np.exp(-d / ((2. * sigma_t) ** 2))
            A[i,j] = A[j,i] = max(s_r * s_t, 1e-8)
    return A

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to configuration file (in JSON)")
    parser.add_argument("task_data", type=str, help="")
    parser.add_argument("task_label", type=str, help="")
    parser.add_argument("ntrain", type=int, help="")
    parser.add_argument("--nrot", type=int, default=3, help="Simulate number of annotation labeled by humans.")
    parser.add_argument("--npos", type=int, default=5, help="Simulate number of annotation labeled by humans.")
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
        config['val_data'] = args.task_data
        config['val_label'] = args.task_label
        config['input_points'] = 30000 # TODO: DELETE ME

    representation, dataset, my_collate_fn, base_model, model, optimizer, loss_function = import_model_by_setting(config, mode='val')
    assert args.ntrain < len(dataset)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=False,
                            shuffle=True,
                            collate_fn=my_collate_fn)
    batch_iterator = iter(dataloader)

    k_dpp = args.nrot
    n_clusters = args.npos
    mcmc_trials = 5
    mcmc_iters = 300

    # Prepare batches
    if args.ntrain > 0:
        print("Preparing training batches...")
        for b in range(args.ntrain):
            batch = next(batch_iterator)
            pc, hand_poses = batch[:2]
            pc_npy = pc[0].numpy().astype(np.float32)
            poses = hand_poses[0]
            kmeans = KMeans(n_clusters=n_clusters, max_iter=100, n_init=3, verbose=False)
            kmeans.fit(poses[...,3])
            selected_grasps = []
            for l in range(n_clusters):
                poses_l = poses[kmeans.labels_==l]
                A = makeAffinityMatrix(poses_l)
                DPP = FiniteDPP('likelihood', **{'L': A})
                best_samples = []
                best_scores = np.inf
                for _ in range(mcmc_trials):
                    samples = DPP.sample_mcmc_k_dpp(size=k_dpp, nb_iter=mcmc_iters)
                    sim = 0.0
                    for i in range(k_dpp):
                        for j in range(i+1, k_dpp):
                            sim += A[samples[i],samples[j]]
                    if sim < best_scores:
                        best_scores = sim
                        best_samples = samples
                selected_grasps += [ poses_l[i] for i in best_samples ]
            selected_grasps = np.asarray(selected_grasps, dtype=np.float32)

            fig = mlab.figure(bgcolor=(0,0,0), size=(1024, 1024))
            col = (pc_npy[:,2] - pc_npy[:,2].min()) / (pc_npy[:,2].max() - pc_npy[:,2].min()) + 0.33
            mlab.points3d(pc_npy[:,0], pc_npy[:,1], pc_npy[:,2], col, scale_factor=0.0015, scale_mode='none', mode='sphere', colormap='plasma', opacity=1.0, figure=fig)
            for pose in selected_grasps:
                gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
                gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
                center_bottom = (gripper_l+gripper_r) / 2.0
                approach = gripper_l_t-gripper_l
                approach = approach / np.linalg.norm(approach) # norm must > 0
                wrist_center = center_bottom - approach * 0.05
                mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=0.003, color=(0, 1, 0), opacity=0.8, figure=fig)
                mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=0.003, color=(0, 1, 0), opacity=0.8, figure=fig)
                mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=0.003, color=(0, 1, 0), opacity=0.8, figure=fig)
                mlab.plot3d([center_bottom[0], wrist_center[0]], [center_bottom[1], wrist_center[1]], [center_bottom[2], wrist_center[2]], tube_radius=0.003, color=(0, 1, 0), opacity=0.8, figure=fig)
            mlab.savefig('shot-%d.png'%b, figure=fig)
            mlab.clf()


            '''
            num_annotated = 0
            for pose_ind in np.random.permutation(len(hand_poses[0])):
                if num_annotated >= args.ngrasp:
                    break
                pose = hand_poses[0][pose_ind]
                gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
                enclosed_pts = crop_index(pc_npy, gripper_outer1, gripper_outer2)
                if len(enclosed_pts)==0:
                    continue
                num_annotated += 1
            '''
