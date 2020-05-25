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

from model.settings.edgeconv import config
from model.representation.euler import *
from model.utils import *
from model.dataset import *
from model.detector.edgeconv.backbone import *
from model.detector.edgeconv.loss import *
from model.detector.utils import *

warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)
warnings.filterwarnings('ignore', category=nb.NumbaPerformanceWarning)


if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    pretrain = sys.argv[1]
    pc_path = sys.argv[2]
    n_proposal = int(sys.argv[3])
    n_iteration = int(sys.argv[4])

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


    base_model = EdgeDet(config).cuda()
    print('Num trainable params: %d'%count_parameters(base_model))
    #model = nn.DataParallel(base_model)
    model = base_model

    states = torch.load(pretrain)
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
            if inferenced_pcs>=n_iteration:
                break
            start_single_object = time.time()
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break

            pc, indices, reverse_lookup_index, cloud_id_batch = batch
            pred = model(pc.cuda(), [pt_idx.cuda() for pt_idx in indices]).cpu().numpy()
            pc_npy = pc.cpu().numpy()
            model_inference_ts = time.time()
            pred_poses = representation.retrive_from_feature_volume_batch(pc_npy, reverse_lookup_index, pred, n_output=n_proposal, threshold=-1e4, nms=True)
            pred_poses = representation.filter_out_invalid_grasp_batch(pc_npy, pred_poses)
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
