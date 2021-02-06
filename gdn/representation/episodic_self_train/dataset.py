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


class GraspDataset(Dataset):
    def __init__(self, config, verbose=True, **kwargs):
        super(GraspDataset, self).__init__(**kwargs)
        self.config = config
        self.max_npts = 30000

        self.train_scene_path = config['train_data']
        self.val_scene_path = config['val_data']

        self.train_ids = list(sorted(list(map(lambda x: os.path.split(x)[-1], glob.glob(self.train_scene_path+'/*.npy')))))
        self.val_ids = list(sorted(list(map(lambda x: os.path.split(x)[-1], glob.glob(self.val_scene_path+'/*.npy')))))

        self.use_aug = False
        self.training = False

    def get_config(self):
        return self.config

    def train(self):
        self.scene_path = self.train_scene_path
        self.ids = self.train_ids
        self.use_aug = True
        self.training = True
    def eval(self):
        self.scene_path = self.val_scene_path
        self.ids = self.val_ids
        self.use_aug = False
        self.training = False
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, item_idx):
        mode = "train" if self.training else "val"
        id_ = self.ids[item_idx]
        # Check if the pre-computed results exist
        config = self.config
        pc_scene = np.load(self.scene_path + '/' + id_) # (#npt, 3)

        # Normalize
        pc_origin = (pc_scene.max(axis=0) + pc_scene.min(axis=0)) / 2.0
        pc_origin[2] = pc_scene[:,2].min() # 0?
        pc_origin = pc_origin.reshape(1, 3)
        pc_scene -= pc_origin
        view_point = np.array([[0.0, 0.0, 2.0]], dtype=np.float32) # (1, 3)

        if self.use_aug:
            # Random translation
            translation = (np.random.randn(3) * 0.03).reshape(1, 3)
            pc_scene += translation

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

        pc_distorted = np.copy(pc_scene)

        if len(pc_distorted)>self.max_npts:
            idx = np.random.choice(len(pc_distorted), self.max_npts, replace=False)
            pc_distorted = pc_distorted[idx]

        if self.use_aug:
            # Random jittering
            pc_distorted += np.random.randn(*pc_distorted.shape) * np.random.uniform(0.000, 0.0025) # noise: 0cm ~ 0.5cm

            # Add isolated points
            n_outliers = np.random.randint(int(np.ceil(0.05 * len(pc_distorted))))
            if n_outliers > 0:
                isolated_points = np.random.uniform(np.min(pc_distorted, axis=0, keepdims=True),
                                  np.max(pc_distorted, axis=0, keepdims=True),
                                  size=(n_outliers, 3))
                pc_distorted = np.append(pc_distorted, isolated_points, axis=0)

        # Simulate single view by Hidden Point Removal
        visible_ind = HPR(pc_distorted, view_point)
        pc_distorted = pc_distorted[visible_ind]

        # Subsample
        while len(pc_distorted)<config['input_points']:
            idx = np.random.choice(len(pc_distorted), min(len(pc_distorted), config['input_points']-len(pc_distorted)), replace=False)
            add_points = pc_distorted[idx] + np.random.randn(len(idx), 3) * 0.5 * 0.001
            pc_distorted = np.unique(np.append(pc_distorted, add_points, axis=0), axis=0)
        if len(pc_distorted)>config['input_points']:
            idx = np.random.choice(len(pc_distorted), config['input_points'], replace=False)
            pc_distorted = pc_distorted[idx]

        return pc_distorted, pc_scene, pc_origin, id_
