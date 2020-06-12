import sys
import os
import re
import gc
import math
import json
import time
import warnings
import random
import numpy as np
import numba as nb
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import copy
from itertools import islice, chain
import pickle
from ...utils import *


class GraspDataset(Dataset):
    def __init__(self, config, **kwargs):
        super(GraspDataset, self).__init__(**kwargs)
        self.config = config

        self.object_set_train = config['object_list']['train']
        self.object_set_val   = config['object_list']['val']
        assert len(set(self.object_set_val) & set(self.object_set_train)) == 0
        self.object_set = self.object_set_train + self.object_set_val

        self.labels = {}
        print('Loading labels from object set...')
        for obj in self.object_set:
            grasp_path = config['grasp_path'] + '/' + obj + '_nms.npy'
            self.labels[obj] = np.load(grasp_path).astype(np.float32)

        self.obj2pcset = {}
        for obj in self.object_set:
            self.obj2pcset[obj] = set(glob.glob(config['point_cloud_path']+'/'+obj+'/clouds/pc_*.npy'))

        self.train_data = []
        for obj in self.object_set_train:
            for pc in self.obj2pcset[obj]:
                self.train_data.append((obj,pc))

        self.val_data = []
        for obj in self.object_set_val:
            for pc in self.obj2pcset[obj]:
                self.val_data.append((obj,pc))

        self.use_aug = False

    def train(self):
        self.data = self.train_data
        self.use_aug = True
    def eval(self):
        self.data = self.val_data
        self.use_aug = False
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item_idx):
        config = self.config

        while True:
            obj_id, cloud_path = self.data[item_idx]
            try:
                vertices = np.load(cloud_path).astype(np.float32)
                if config['full_view']:
                    other_views = list(self.obj2pcset[obj_id] - set([cloud_path]))
                    random.shuffle(other_views)
                    num_views = np.random.randint(config['min_view'], config['max_view']+1)
                    view_count = 1
                    for pc_path in other_views:
                        if view_count >= num_views:
                            break
                        try:
                            other_pc = np.load(pc_path).astype(np.float32)
                        except Exception as e:
                            warnings.warn('Failed to read a npy file. Is \"%s\" broken? : %s'%(pc_path, str(e)))
                            continue
                        if len(other_pc.shape)!=2 or other_pc.shape[0]<=0: # no points
                            continue
                        vertices = np.append(vertices, other_pc, axis=0)
                        view_count += 1
                if len(vertices)==0:
                    raise ValueError("Empty point cloud!")
            except Exception as e:
                warnings.warn('Failed to read a npy file. Is \"%s\" broken? : %s'%(cloud_path, str(e)))
                item_idx = np.random.randint(len(self.data))
                continue
            break
        # Padding for point clouds
        vertices = np.unique(vertices, axis=0) # FIXME: O(NlogN)
        while len(vertices)<config['input_points']:
            idx = np.random.choice(len(vertices), min(len(vertices), config['input_points']-len(vertices)), replace=False)
            add_points = vertices[idx] + np.random.randn(len(idx), 3) * 0.5 * 0.001
            vertices = np.unique(np.append(vertices, add_points, axis=0), axis=0)
        if len(vertices)>config['input_points']:
            idx = np.random.choice(len(vertices), config['input_points'], replace=False)
            vertices = vertices[idx]
        np.random.shuffle(vertices)

        use_aug = self.use_aug and np.random.rand()<0.5

        if use_aug:
            ### Do preprocessing, augmentation here ###
            random_rotation = random_rotation_matrix_xyz()
            vertices = np.matmul(vertices, random_rotation.T)

        ### Read & Decode Hand Pose ###
        hand_poses = np.copy(self.labels[obj_id])
        if len(hand_poses)>config['max_sample_grasp_per_object']:
            hand_poses = hand_poses[np.random.choice(len(hand_poses),
                                                     config['max_sample_grasp_per_object'],
                                                     replace=False)]
        else: # <=
            np.random.shuffle(hand_poses)
        pose_idx_pairs = []
        for pose in hand_poses:
            ### Do preprocessing, augmentation here ###
            gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])

            if use_aug:
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

            enclosed_pts = crop_index(vertices, gripper_outer1, gripper_outer2)
            if len(enclosed_pts)==0:
                continue
            pose_idx_pairs.append((pose, set(enclosed_pts)))

        return vertices, pose_idx_pairs

class collate_fn_setup(object):
    def __init__(self, config, representation):
        self.config = config
        self.representation = representation
        self.aug = True
    def train(self):
        self.aug = True
    def eval(self):
        self.aug = False
    def __call__(self, batch):
        config = self.config
        representation = self.representation
        pc_batch, pose_idx_pairs_batch = zip(*batch)
        pc_batch = np.stack(pc_batch).astype(np.float32)
        pose_idx_pairs_batch = list(pose_idx_pairs_batch)

        if self.aug and np.random.rand()<0.3:
            preserve_npts = int(np.round(np.random.uniform(pc_batch.shape[1]*(1-self.config['input_dropout_rate']), pc_batch.shape[1])))
            preserve_npts = max(self.config['subsample_levels'][0], min(pc_batch.shape[1], preserve_npts))
            pc_batch = pc_batch[:,:preserve_npts,:]

        network_indices = []
        # featurefeature_volume_batch shape: (B, M, 7): -> logit, xyz(3), rpy(3)
        feature_volume_batch = []

        input_points = pc_batch.shape[1]
        prev_index = np.stack([np.arange(input_points, dtype=np.int32) for _ in range(len(pc_batch))])
        reverse_lookup_index = [np.arange(input_points, dtype=np.int32) for _ in range(len(pc_batch))]

        prev_index = np.repeat(prev_index[...,np.newaxis], 3, axis=-1)
        output_index = None

        for layer, n_pts in enumerate(config['subsample_levels']):
            index = np.stack([np.random.choice(prev_index.shape[1], n_pts, replace=False) for _ in range(len(pc_batch))])
            network_indices.append(torch.IntTensor(index))
            prev_index = np.take_along_axis(prev_index, np.repeat(index[...,np.newaxis], 3, -1), axis=1)

            if layer<=config['output_layer']: # only need to lookup to output indices
                for b, ind in enumerate(index):
                    reverse_lookup_index[b] = reverse_lookup_index[b][ind] # (N,) -> (N',)
                if layer==config['output_layer']:
                    output_index = prev_index[:,:,0] # for generate feature: (B, N')
            # else: # Network can learn global feature
        assert output_index is not None

        # Fix indices by lookup reverse lookup indices
        batch_gt_poses = []
        for b in range(len(pose_idx_pairs_batch)):
            reverse_index = reverse_lookup_index[b] # (N')
            sample_dict = {}
            for n, (pose, ind) in enumerate(pose_idx_pairs_batch[b]):
                new_ind = index_to_new_index(ind, reverse_index) # indices in subsampled points
                if len(new_ind)==0:
                    continue
                xyz, rpy = representation.grasp_representation(pose,
                               pc_batch[b,output_index[b],:], # shape: (N', 3), subsampled
                               new_ind)
                unique_grasp_key = tuple(new_ind)
                unique_grasp_value = (xyz, rpy)
                sample_dict[unique_grasp_key] = unique_grasp_value
                if len(sample_dict)>=config['max_grasp_per_object']:
                    break
            feature_volume = np.zeros((config['subsample_levels'][config['output_layer']],
                                       7), dtype=np.float32)
            for n, (new_ind, (xyz, rpy)) in enumerate(sample_dict.items()):
                representation.update_feature_volume(feature_volume, new_ind, xyz, rpy)
            feature_volume_batch.append(feature_volume)
            if len(pose_idx_pairs_batch[b])>0:
                batch_gt_poses.append( next(zip(*pose_idx_pairs_batch[b]))  )
        return torch.FloatTensor(pc_batch), torch.FloatTensor(feature_volume_batch), network_indices, reverse_lookup_index, batch_gt_poses


class GraspDatasetVal(Dataset):
    def __init__(self, config, **kwargs):
        super(GraspDatasetVal, self).__init__(**kwargs)
        self.config = config
        assert 'object_list' in config

        self.object_set_val   = config['object_list']['val']

        self.data = []
        for obj in self.object_set_val:
            for pc in glob.glob(config['point_cloud_path']+'/'+obj+'/clouds/*.npy'):
                self.data.append((obj,pc))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item_idx):
        config = self.config

        while True:
            obj_id, cloud_path = self.data[item_idx]
            cloud_id = os.path.splitext(cloud_path)[0].split('/')
            cloud_id = (cloud_id[-3], cloud_id[-1])
            try:
                vertices = np.load(cloud_path)
                if len(vertices)==0:
                    raise ValueError("Empty point cloud!")
            except Exception as e:
                warnings.warn('Failed to read a npy file. Is \"%s\" broken? : %s'%(cloud_path, str(e)))
                item_idx = np.random.randint(len(self.data))
                continue
            break

        # Padding for point clouds
        vertices = np.unique(vertices, axis=0) # FIXME: O(NlogN)
        while len(vertices)<config['input_points']:
            idx = np.random.choice(len(vertices), min(len(vertices), config['input_points']-len(vertices)), replace=False)
            add_points = vertices[idx] + np.random.randn(len(idx), 3) * 0.5 * 0.001
            vertices = np.unique(np.append(vertices, add_points, axis=0), axis=0)
        if len(vertices)>config['input_points']:
            idx = np.random.choice(len(vertices), config['input_points'], replace=False)
            vertices = vertices[idx]
        np.random.shuffle(vertices)

        return vertices, cloud_id

class val_collate_fn_setup(object):
    def __init__(self, config):
        self.config = config
    def __call__(self, batch):
        config = self.config
        pc_batch, cloud_id_batch = zip(*batch)
        pc_batch = np.stack(pc_batch).astype(np.float32)

        network_indices = []

        input_points = pc_batch.shape[1]
        prev_index = np.stack([np.arange(input_points, dtype=np.int32) for _ in range(len(pc_batch))])
        reverse_lookup_index = [np.arange(input_points, dtype=np.int32) for _ in range(len(pc_batch))]

        prev_index = np.repeat(prev_index[...,np.newaxis], 3, axis=-1)
        output_index = None

        for layer, n_pts in enumerate(config['subsample_levels']):
            index = np.stack([np.random.choice(prev_index.shape[1], n_pts, replace=False) for _ in range(len(pc_batch))])
            network_indices.append(torch.IntTensor(index))
            prev_index = np.take_along_axis(prev_index, np.repeat(index[...,np.newaxis], 3, -1), axis=1)

            if layer<=config['output_layer']: # only need to lookup to output indices
                for b, ind in enumerate(index):
                    reverse_lookup_index[b] = reverse_lookup_index[b][ind] # (N,) -> (N',)
                if layer==config['output_layer']:
                    output_index = prev_index[:,:,0] # for generate feature: (B, N')
            # else: # Network can learn global feature
        assert output_index is not None
        return torch.FloatTensor(pc_batch), network_indices, reverse_lookup_index, cloud_id_batch
