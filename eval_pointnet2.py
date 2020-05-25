# coding: utf-8

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
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
from collections import namedtuple
from pointnet2.utils import pointnet2_utils
torch.backends.cudnn.benchmark = True
from tqdm import tqdm
import pickle
from itertools import islice, chain
from torch.utils.data import Dataset, DataLoader
import copy

from model.settings.pointnet2 import config
from model.representation.euler import *
from model.utils import *
from model.dataset import *
from model.detector.pointnet2.backbone import *
from model.detector.pointnet2.loss import *
from model.detector.utils import *

warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)
warnings.filterwarnings('ignore', category=nb.NumbaPerformanceWarning)


if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    pretrain = sys.argv[1]
    pc_path = sys.argv[2]
    output_dir = sys.argv[3]

    config['point_cloud_path'] = pc_path

    representation = EulerRepresentation(config)

    dataset = GraspDatasetVal(config)
    my_collate_fn = val_collate_fn_setup(config)
    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=my_collate_fn)


    base_model = Pointnet2MSG(config).cuda()
    print('Num trainable params: %d'%count_parameters(base_model))
    #model = nn.DataParallel(base_model)
    model = base_model

    states = torch.load(pretrain)
    base_model.load_state_dict(states['base_model'])

    model.eval()
    batch_iterator = iter(dataloader)
    with torch.no_grad():
        while True:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break
            pc, indices, reverse_lookup_index, cloud_id_batch = batch
            pred = model(pc.cuda(), [pt_idx.cuda() for pt_idx in indices]).cpu().numpy()
            pc_npy = pc.cpu().numpy()
            pred_poses = representation.retrive_from_feature_volume_batch(pc_npy, reverse_lookup_index, pred, n_output=300, threshold=-1e8, nms=True)
            pred_poses = representation.filter_out_invalid_grasp_batch(pc_npy, pred_poses)
            for pose, cloud_id in zip(pred_poses, cloud_id_batch):
                prefix = output_dir + '/' + cloud_id[0]
                if not os.path.exists(prefix):
                    os.makedirs(prefix)
                if len(pose)>0:
                    lscore, lpose_3x4 = zip(*pose)
                else:
                    lscore, lpose_3x4 = [], []
                meta = [(lscore[n], cloud_id[0], cloud_id[1], n) for n in range(len(lscore))]
                with open(prefix+'/'+cloud_id[1]+'.meta', 'wb') as fp:
                    pickle.dump(meta, fp, protocol=2, fix_imports=True)
                lpose_3x4 = np.asarray(lpose_3x4, dtype=np.float32)
                np.save(prefix+'/'+cloud_id[1]+'.npy', lpose_3x4)
                print('Processed: {}_{} {:s}'.format(cloud_id[0], cloud_id[1], str(lpose_3x4.shape)))

