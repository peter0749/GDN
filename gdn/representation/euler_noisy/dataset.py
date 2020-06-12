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

        vertices = vertices + np.random.randn(3).reshape(1, 3) * config['noise_std_cm'] * 0.01

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

        # Check if model support the number of points
        while len(vertices)<config["subsample_levels"][0]:
            # Too few points. Just replicate points to match model input shape.
            idx = np.random.choice(len(vertices), min(len(vertices), config["subsample_levels"][0]-len(vertices)), replace=False)
            add_points = vertices[idx] + np.random.randn(len(idx), 3) * 0.5 * 0.001
            vertices = np.unique(np.append(vertices, add_points, axis=0), axis=0)

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
