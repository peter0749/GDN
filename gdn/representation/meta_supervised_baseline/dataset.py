import sys
import os
import re
import math
import json
import time
import warnings
import random
import uuid
import numpy as np
import numba as nb
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import copy
from itertools import islice, chain
#import pickle
import h5py
from ...utils import *

# import pcl # TODO: Delete Me


class GraspDataset(Dataset):
    def __init__(self, config, verbose=True, **kwargs):
        super(GraspDataset, self).__init__(**kwargs)
        self.config = config
        self.max_npts = 30000

        self.scene_path = config['train_data']
        self.label_path = config['train_label']
        self.labels = list(sorted(list(map(lambda x: '/'.join(x.split('/')[-2:]), glob.glob(self.label_path+'/*/*.npy')))))

        self.use_aug = False
        self.training = False

    def get_config(self):
        return self.config

    def train(self):
        self.use_aug = True
        self.training = True
    def eval(self):
        self.use_aug = False
        self.training = False
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item_idx):
        id_ = self.labels[item_idx]
        config = self.config
        pc_scene = np.load(self.scene_path + '/' + id_).astype(np.float32) # (#npt, 3)
        hand_poses = np.load(self.label_path + '/' + id_).astype(np.float32) # (N, 3, 4)

        if len(pc_scene)>self.max_npts:
            idx = np.random.choice(len(pc_scene), self.max_npts, replace=False)
            pc_scene = pc_scene[idx]

        # Normalize
        pc_origin = (pc_scene.max(axis=0) + pc_scene.min(axis=0)) / 2.0
        pc_origin[2] = pc_scene[:,2].min() # 0?
        pc_origin = pc_origin.reshape(1, 3)
        pc_scene -= pc_origin
        if len(hand_poses) > 0:
            hand_poses[:,:,3] -= pc_origin
        view_point = np.array([[0.0, 0.0, 2.0]], dtype=np.float32) # (1, 3)

        if self.use_aug:
            # Random translation
            translation = (np.random.randn(3) * 0.03).reshape(1, 3)
            pc_scene += translation
            if len(hand_poses) > 0:
                hand_poses[:,:,3] += translation

            # Random rotation
            rx = float(np.random.uniform(-0.10, 0.10)) # ~8 degrees
            ry = float(np.random.uniform(-0.10, 0.10)) # ~8 degrees
            rz = float(np.random.uniform(-np.pi, np.pi))
            random_rotation = rotation_euler(rx, ry, rz)
            pc_scene = np.matmul(pc_scene, random_rotation.T)

            # Random viewpoint
            view_point[0,0] = np.random.uniform(-0.4, 0.4) # -+ 40cm
            view_point[0,1] = np.random.uniform(-0.4, 0.4) # -+ 40cm
            view_point[0,2] = np.random.uniform(0.6, 3.0) # 0.6m ~ 3.0m

            # Random jittering
            pc_scene += np.random.randn(*pc_scene.shape) * np.random.uniform(0.000, 0.0025) # noise: 0cm ~ 0.5cm

            # Add isolated points
            n_outliers = np.random.randint(int(np.ceil(0.05 * len(pc_scene))))
            if n_outliers > 0:
                isolated_points = np.random.uniform(np.min(pc_scene, axis=0, keepdims=True),
                                  np.max(pc_scene, axis=0, keepdims=True),
                                  size=(n_outliers, 3))
                pc_scene = np.append(pc_scene, isolated_points, axis=0)

        # Simulate single view by Hidden Point Removal
        visible_ind = HPR(pc_scene, view_point)
        pc_scene = pc_scene[visible_ind]
        # Save to PLY for visualization?
        # TODO: Delete ME
        # pc = pcl.PointCloud(pc_scene)
        # pcl.save(pc, './'+id_[:-4]+'.pcd')
        # del pc

        # Subsample
        # pc_scene = np.unique(pc_scene, axis=0) # FIXME: O(NlogN)
        while len(pc_scene)<config['input_points']:
            idx = np.random.choice(len(pc_scene), min(len(pc_scene), config['input_points']-len(pc_scene)), replace=False)
            add_points = pc_scene[idx] + np.random.randn(len(idx), 3) * 0.5 * 0.001
            pc_scene = np.unique(np.append(pc_scene, add_points, axis=0), axis=0)
        if len(pc_scene)>config['input_points']:
            idx = np.random.choice(len(pc_scene), config['input_points'], replace=False)
            pc_scene = pc_scene[idx]

        ### Read & Decode Hand Pose ###
        if len(hand_poses)>config['max_sample_grasp']:
            hand_poses = hand_poses[np.random.choice(len(hand_poses),
                                                     config['max_sample_grasp'],
                                                     replace=False)]
        else: # <=
            np.random.shuffle(hand_poses)
        poses_list = None
        enclosed_pts_list = []
        for pose in hand_poses:
            ### Do preprocessing, augmentation here ###
            gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])

            if self.use_aug:
                gripper_inner_edge = np.matmul(gripper_inner_edge, random_rotation.T)
                gripper_outer1 = np.matmul(gripper_outer1, random_rotation.T)
                gripper_outer2 = np.matmul(gripper_outer2, random_rotation.T)

                bottom_center = (gripper_inner_edge[0] + gripper_inner_edge[1]) / 2.0 # (3,)
                tip_center = (gripper_inner_edge[2] + gripper_inner_edge[3]) / 2.0 # (3,)
                bottom_right = gripper_inner_edge[1]
                x_axis = tip_center-bottom_center
                x_axis = x_axis / np.linalg.norm(x_axis, ord=2)
                y_axis = bottom_right-bottom_center
                y_axis = y_axis / np.linalg.norm(y_axis, ord=2)
                z_axis = np.cross(x_axis, y_axis)
                #z_axis = z_axis / np.linalg.norm(z_axis, ord=2)
                pose = np.concatenate([
                    x_axis[...,np.newaxis],
                    y_axis[...,np.newaxis],
                    z_axis[...,np.newaxis],
                    bottom_center[...,np.newaxis]
                ], axis=1) # (3,4)

            #assert np.abs(np.linalg.det(pose[:3,:3])-1.0) < 1e-5

            enclosed_pts = crop_index(pc_scene, gripper_outer1, gripper_outer2)
            if len(enclosed_pts)==0:
                continue
            if poses_list is None:
                poses_list = pose[np.newaxis]
            else:
                poses_list = np.append(poses_list, pose[np.newaxis], axis=0) # (N, 3, 4)
            enclosed_pts_list.append(np.array(list(set(enclosed_pts)), dtype=np.int32)) # (N,)

        return pc_scene, poses_list, enclosed_pts_list

class collate_fn_setup(object):
    def __init__(self, config, representation):
        self.config = config
        self.representation = representation
    def train(self):
        pass
    def eval(self):
        pass
    def __call__(self, batch):
        config = self.config
        representation = self.representation
        pc_batch, pose_batch, enclosed_pt_batch = zip(*batch)
        pc_batch = np.stack(pc_batch).astype(np.float32)
        input_points = pc_batch.shape[1]
        feature_volume_batch = np.zeros((pc_batch.shape[0], input_points,
                                       config['n_pitch'],
                                       config['n_yaw'],
                                       8), dtype=np.float32)
        for b in range(pc_batch.shape[0]):
            if pose_batch[b] is None or len(pose_batch[b]) == 0 or \
               enclosed_pt_batch[b] is None or len(enclosed_pt_batch[b]) == 0:
                continue
            for n, (pose, ind) in enumerate(zip(pose_batch[b], enclosed_pt_batch[b])):
                if len(ind)==0:
                    continue
                (xyz,
                 roll,
                 pitch_index,
                 pitch_residual,
                 yaw_index,
                 yaw_residual) = representation.grasp_representation(pose,
                 pc_batch[b,:], # shape: (N, 3)
                 ind)
                representation.update_feature_volume(feature_volume_batch[b], ind, xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual)
        return torch.FloatTensor(pc_batch), torch.FloatTensor(feature_volume_batch), pose_batch

class GraspDatasetVal(Dataset):
    def __init__(self, config, **kwargs):
        super(GraspDatasetVal, self).__init__(**kwargs)
        self.config = config
        self.max_npts = 30000
        self.val_scene_path = config['val_data']
        self.val_label_path = config['val_label']
        self.val_labels = list(map(lambda x: os.path.split(x)[-1], glob.glob(self.val_label_path+'/*.npy')))
        self.use_aug = False
        self.scene_path = self.val_scene_path
        self.label_path = self.val_label_path
        self.labels = self.val_labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item_idx):
        config = self.config
        id_ = self.labels[item_idx]
        pc_scene = np.load(self.scene_path + '/' + id_) # (#npt, 3)
        hand_poses = np.load(self.label_path + '/' + id_) # (N, 3, 4)

        # Normalize
        pc_origin = (pc_scene.max(axis=0) + pc_scene.min(axis=0)) / 2.0
        pc_origin[2] = pc_scene[:,2].min() # 0?
        pc_origin = pc_origin.reshape(1, 3)
        pc_scene -= pc_origin
        hand_poses[:,:,3] -= pc_origin
        view_point = np.array([[0.0, 0.0, 2.0]], dtype=np.float32) # (1, 3)

        if len(pc_scene)>self.max_npts:
            idx = np.random.choice(len(pc_scene), self.max_npts, replace=False)
            pc_scene = pc_scene[idx]

        # Simulate single view by Hidden Point Removal
        visible_ind = HPR(pc_scene, view_point)
        pc_scene = pc_scene[visible_ind]
        # Save to PLY for visualization?
        # TODO: Delete ME
        # pc = pcl.PointCloud(pc_scene)
        # pcl.save(pc, "simulated_single_view.ply", format='ply')

        # Subsample
        # pc_scene = np.unique(pc_scene, axis=0) # FIXME: O(NlogN)
        while len(pc_scene)<config['input_points']:
            idx = np.random.choice(len(pc_scene), min(len(pc_scene), config['input_points']-len(pc_scene)), replace=False)
            add_points = pc_scene[idx] + np.random.randn(len(idx), 3) * 0.5 * 0.001
            pc_scene = np.unique(np.append(pc_scene, add_points, axis=0), axis=0)
        if len(pc_scene)>config['input_points']:
            idx = np.random.choice(len(pc_scene), config['input_points'], replace=False)
            pc_scene = pc_scene[idx]

        ### Read & Decode Hand Pose ###
        if len(hand_poses)>config['max_sample_grasp']:
            hand_poses = hand_poses[np.random.choice(len(hand_poses),
                                                     config['max_sample_grasp'],
                                                     replace=False)]
        else: # <=
            np.random.shuffle(hand_poses)

        return pc_scene, hand_poses, pc_origin, os.path.splitext(id_)[0]

class collate_fn_setup_val(object):
    def __init__(self, config):
        self.config = config
        self.aug = False
    def __call__(self, batch):
        config = self.config
        pc_batch, batch_gt_poses, pc_origin, scene_id = zip(*batch)
        pc_batch = np.stack(pc_batch).astype(np.float32)
        pc_origin = np.stack(pc_origin) # (N, 1, 3)
        return torch.FloatTensor(pc_batch), batch_gt_poses, pc_origin, scene_id
