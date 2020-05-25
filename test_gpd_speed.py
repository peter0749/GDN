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

from pointnet import PointNetCls, DualPointNetCls
from gpd import GPDClassifier

from collections import namedtuple
torch.backends.cudnn.benchmark = True
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from pykdtree.kdtree import KDTree
import point_cloud_utils as pcu
from itertools import islice, chain
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler
import copy
import multiprocessing

warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)
warnings.filterwarnings('ignore', category=nb.NumbaPerformanceWarning)

# In[2]:


config = dict(
    sample_path = '',
    point_cloud_path = '',
    eval_ideal_sampler = False,
    hand_height = 0.11, # 11cm (x range)
    gripper_width = 0.08, # 8cm (y range)
    thickness_side = 0.01, # (z range)
    thickness = 0.01, # gripper_width + thickness*2 = hand_outer_diameter (0.08+0.01*2 = 0.1)
    trans_th=0.02,
    rot_th=5.0,
    input_points=1000,
    min_point_limit=5,
    batch_size=128,
    eval_split=0.2,
    object_list = {
        "train": ["077_rubiks_cube", "065-b_cups", "007_tuna_fish_can", "026_sponge", "011_banana", "021_bleach_cleanser", "016_pear", "014_lemon", "061_foam_brick", "004_sugar_box", "057_racquetball", "065-a_cups", "019_pitcher_base", "048_hammer", "015_peach", "009_gelatin_box", "056_tennis_ball", "071_nine_hole_peg_test", "065-c_cups", "024_bowl", "065-f_cups", "043_phillips_screwdriver", "003_cracker_box", "065-g_cups", "035_power_drill", "055_baseball", "065-d_cups", "029_plate", "033_spatula", "006_mustard_bottle", "052_extra_large_clamp"],
        "val": ["008_pudding_box", "065-e_cups", "037_scissors", "018_plum", "010_potted_meat_can", "017_orange", "005_tomato_soup_can"]
    }
    )

@nb.jit(nopython=True, nogil=True)
def hand_match(pred, target, rot_th=5, trans_th=0.02):
    # pred: (3, 4)
    # target: (3, 4)
    rot_th_rad = np.radians(rot_th)
    trans_matched = np.linalg.norm(pred[:,3]-target[:,3], ord=2) < trans_th
    rot_diff = np.arcsin(np.linalg.norm( np.eye(3) - np.dot(pred[:3,:3], target[:3,:3].T), ord=None  ) / (2*np.sqrt(2)))
    rot_matched = rot_diff < rot_th_rad
    return rot_matched and trans_matched

class GraspDatasetGPD(object):
    def __init__(self, config, projection=False, project_chann=3, project_size=60, max_candidate=2140000000, **kwargs):
        super(GraspDatasetGPD, self).__init__(**kwargs)
        self.config = config

        cloud_set = set(filter(lambda x: re.match(r'^[0-9]{3}(-|_)[0-9a-zA-Z]+.*$', x) and                               len(glob.glob(config['point_cloud_path']+'/'+x+'/clouds/*.npy'))>0
                    , os.listdir(config['point_cloud_path'])))
        self.object_set = list(cloud_set)

        self.projection = projection
        self.project_chann = project_chann
        if self.project_chann not in [3, 12]:
            raise NotImplementedError
        self.project_size = project_size
        if self.project_size != 60:
            raise NotImplementedError
        self.max_candidate = max_candidate
        self.normal_K = 10
        self.voxel_point_num  = 50
        self.projection_margin = 1
        self.minimum_point_amount = 150

        if 'object_list' in config:
            self.object_set_train = config['object_list']['train']
            self.object_set_val   = config['object_list']['val']
        else:
            random.shuffle(self.object_set)
            num_test = int(np.ceil(config['eval_split']*len(self.object_set)))
            self.object_set_val = self.object_set[:num_test]
            self.object_set_train = self.object_set[num_test:]
        assert len(set(self.object_set_val) & set(self.object_set_train)) == 0

        self.train_data = []
        for obj in self.object_set_train:
            for pc in glob.glob(config['point_cloud_path']+'/'+obj+'/clouds/*.npy'):
                self.train_data.append((obj,pc))

        self.val_data = []
        for obj in self.object_set_val:
            for pc in glob.glob(config['point_cloud_path']+'/'+obj+'/clouds/*.npy'):
                self.val_data.append((obj,pc))

    def cal_projection(self, point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, gripper_width):
        occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1])
        norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3])
        norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1])

        max_x = point_cloud_voxel[:, order[0]].max()
        min_x = point_cloud_voxel[:, order[0]].min()
        max_y = point_cloud_voxel[:, order[1]].max()
        min_y = point_cloud_voxel[:, order[1]].min()
        min_z = point_cloud_voxel[:, order[2]].min()

        tmp = max((max_x - min_x), (max_y - min_y))
        if tmp == 0:
            return occupy_pic, norm_pic
        # Here, we use the gripper width to cal the res:
        res = gripper_width / (m_width_of_pic-margin)

        voxel_points_square_norm = []
        x_coord_r = ((point_cloud_voxel[:, order[0]]) / res + m_width_of_pic / 2)
        y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
        z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
        x_coord_r = np.floor(x_coord_r).astype(int)
        y_coord_r = np.floor(y_coord_r).astype(int)
        z_coord_r = np.floor(z_coord_r).astype(int)
        voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
        coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
        K = len(coordinate_buffer)
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=K, dtype=np.int64)
        feature_buffer = np.zeros(shape=(K, self.voxel_point_num, 6), dtype=np.float32)
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

        for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < self.voxel_point_num:
                feature_buffer[index, number, :3] = point
                feature_buffer[index, number, 3:6] = normal
                number_buffer[index] += 1

        voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1)/number_buffer[:, np.newaxis]
        voxel_points_square = coordinate_buffer

        if len(voxel_points_square) == 0:
            return occupy_pic, norm_pic
        x_coord_square = voxel_points_square[:, 0]
        y_coord_square = voxel_points_square[:, 1]
        norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
        occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
        occupy_max = occupy_pic.max()
        assert(occupy_max > 0)
        occupy_pic = occupy_pic / occupy_max
        return occupy_pic, norm_pic

    def project_pc(self, pc, gripper_width, in_ind):
        """
        for gpd baseline, only support input_chann == [3, 12]
        """
        pc = pc.astype(np.float32)
        '''
        pc = pcl.PointCloud(pc)
        norm = pc.make_NormalEstimation()
        norm.set_KSearch(self.normal_K)
        normals = norm.compute()
        surface_normal = normals.to_array()
        '''
        surface_normal = pcu.estimate_normals(pc, k=self.normal_K)
        surface_normal = surface_normal[:, 0:3]

        grasp_pc = pc[in_ind]
        grasp_pc_norm = surface_normal[in_ind]
        bad_check = (grasp_pc_norm != grasp_pc_norm)
        if np.sum(bad_check)!=0:
            bad_ind = np.where(bad_check == True)
            grasp_pc = np.delete(grasp_pc, bad_ind[0], axis=0)
            grasp_pc_norm = np.delete(grasp_pc_norm, bad_ind[0], axis=0)
        assert(np.sum(grasp_pc_norm != grasp_pc_norm) == 0)
        m_width_of_pic = self.project_size
        margin = self.projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
        if self.project_chann == 3:
            output = norm_pic1
        elif self.project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                         order, gripper_width)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = self.cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm,
                                                     order, gripper_width)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError

        return output

    def collect_pc(self, pc, pose):
        width = self.config['gripper_width']
        matrix = pose[:3,:3] # 3x3
        approach = matrix[:,0] # (3,)
        bottom = pose[:3, 3] # (3,) need conversion
        center = bottom + self.config['hand_height'] * approach # compute center between gripper tips
        center = center.reshape(1,3)
        #pc_t = (np.dot(matrix, (pc-center).T)).T
        pc_t = (np.dot(matrix.T, (pc-center).T)).T

        x_limit = width/4
        z_limit = width/4
        y_limit = width/2

        x1 = pc_t[:, 0] > -x_limit
        x2 = pc_t[:, 0] < x_limit
        y1 = pc_t[:, 1] > -y_limit
        y2 = pc_t[:, 1] < y_limit
        z1 = pc_t[:, 2] > -z_limit
        z2 = pc_t[:, 2] < z_limit

        a = np.vstack([x1, x2, y1, y2, z1, z2]) # (6, N)
        in_ind = np.where(np.sum(a, axis=0) == len(a))[0]

        if len(in_ind) < self.config['min_point_limit']:
            return None
        if self.projection:
            return self.project_pc(pc_t, width, in_ind)
        else:
            pc_t = pc_t[in_ind]
            if len(pc_t) < self.config['input_points']:
                new_pt = pc_t[np.random.choice(len(pc_t), self.config['input_points']-len(pc_t), replace=True)]
                pc_t = np.append(pc_t, new_pt, axis=0)
            elif len(pc_t) > self.config['input_points']:
                pc_t = pc_t[np.random.choice(len(pc_t), self.config['input_points'], replace=False)]
            return pc_t

    def train(self):
        self.data = self.train_data
    def eval(self):
        self.data = self.val_data
    def list_objects(self):
        return {
                'train': self.object_set_train,
                'val':   self.object_set_val
               }
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item_idx):
        config = self.config
        obj_id, cloud_path = self.data[item_idx]
        cloud_id = os.path.splitext(cloud_path)[0].split('/')
        cloud_id = (cloud_id[-3], cloud_id[-1])
        grasp_pc = None
        if config['eval_ideal_sampler']:
            candidate = np.load(config['sample_path']+'/'+cloud_id[0]+'.npy')
        else:
            candidate = np.load(config['sample_path']+'/'+cloud_id[0]+'/'+cloud_id[1]+'.npy')
        if len(candidate.shape)!=3: # no sample
            candidate = []
        if len(candidate)>self.max_candidate:
            candidate = candidate[np.random.choice(len(candidate), self.max_candidate, replace=False)]
        vertices = np.load(cloud_path)
        def gen():
            pool = multiprocessing.Pool(processes=8)
            res = [ pool.apply_async(self.collect_pc, args=(vertices, pose)) for pose in candidate ]
            for i in range(len(res)):
                grasp_pc_i = res[i].get()
                yield grasp_pc_i
            pool.close()
            pool.join()
            del pool
        return cloud_id, candidate, gen

if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    method = sys.argv[1]
    pretrain_path = sys.argv[2]
    sample_path = sys.argv[3]
    point_cloud_path = sys.argv[4]
    max_candidate = int(sys.argv[5])
    n_iteration = int(sys.argv[6])

    config['sample_path'] = sample_path
    config['point_cloud_path'] = point_cloud_path

    if method == 'GPD':
        dataset = GraspDatasetGPD(config, projection=True, project_chann=3, project_size=60, max_candidate=max_candidate)
        model = GPDClassifier(3)
    elif method == 'PointNetGPD':
        dataset = GraspDatasetGPD(config, projection=False, project_chann=3, project_size=60, max_candidate=max_candidate)
        model = PointNetCls(num_points=config['input_points'], input_chann=3, k=2)
    elif method == 'PointNetGPDMC':
        dataset = GraspDatasetGPD(config, projection=False, project_chann=3, project_size=60, max_candidate=max_candidate)
        model = PointNetCls(num_points=config['input_points'], input_chann=3, k=3)
    else:
        raise NotImplementedError

    state = torch.load(pretrain_path)
    model.load_state_dict(state, strict=False)
    model = model.cuda()

    model.eval()
    dataset.eval()

    elapsed_time = 0.0
    nms_time_avg = 0.0
    model_inference_time_avg = 0.0
    single_object_inference_time_avg = 0.0

    start_ts = time.time()
    single_object_start_ts = time.time()

    with torch.no_grad():
        for inferenced_pcs, (cloud_id, candidate, generator) in enumerate(dataset):
            if inferenced_pcs>=n_iteration:
                break # Stop the benchmark
            logits = np.full(len(candidate), -np.inf)
            gen = generator()
            batch = []
            batch_idx = []
            for i, pc in enumerate(gen):
                if pc is None or len(pc)==0: # SKIP, no points / invalid input
                    continue
                if len(pc.shape)==2:   # point cloud (B, N, C) -> (B, C, N)
                    inp = np.transpose(pc[np.newaxis], (0, 2, 1))
                elif len(pc.shape)==3: # multi view CNN (B, H, W, C) -> (B, C, H, W)
                    inp = np.transpose(pc[np.newaxis], (0, 3, 1, 2))
                else:
                    raise NotImplementedError("Invalid Input Shape!!!")
                batch.append(inp)
                batch_idx.append(i)
                if len(batch)>=config['batch_size']:
                    out = model(torch.from_numpy(np.concatenate(batch, axis=0)).float().cuda())
                    if isinstance(out, tuple):
                        out = out[0][:,-1].cpu().numpy() # (#batch,)
                    else:
                        out = out[:,-1].cpu().numpy() # (#batch,)
                    logits[batch_idx] = out
                    batch = []
                    batch_idx = []
            if len(batch)>0:
                out = model(torch.from_numpy(np.concatenate(batch, axis=0)).float().cuda())
                if isinstance(out, tuple):
                    out = out[0][:,-1].cpu().numpy() # (#batch,)
                else:
                    out = out[:,-1].cpu().numpy() # (#batch,)
                logits[batch_idx] = out
            del batch, batch_idx, gen
            candidate = np.asarray(candidate, dtype=np.float32)
            # Filter out poses that cannot be encoded by model
            # Since in real-world we will discard them too...
            valid = (logits!=-np.inf)
            logits = logits[valid]
            candidate = candidate[valid]
            rank = np.argsort(-logits)
            logits = logits[rank]
            candidate = candidate[rank]

            nms_start_ts = time.time()
            # NMS
            matched_index = set()
            for n,pose in enumerate(candidate):
                matched = False
                for i in matched_index:
                    if hand_match(pose, candidate[i], rot_th=config['rot_th'], trans_th=config['trans_th']):
                        matched = True
                        break
                if not matched:
                    matched_index.add(n)
            matched_index = list(matched_index)
            matched_index.sort()
            matched_index = np.asarray(matched_index, dtype=np.int32)
            candidate = candidate[matched_index]
            logits = logits[matched_index]
            single_object_end_ts = time.time()

            nms_time = single_object_end_ts - nms_start_ts
            model_inference_time = nms_start_ts - single_object_start_ts
            single_object_inference_time = single_object_end_ts - single_object_start_ts

            nms_time_avg += nms_time
            model_inference_time_avg += model_inference_time
            single_object_inference_time_avg += single_object_inference_time

            single_object_start_ts = time.time()
        end_ts = time.time()

        elapsed_time = end_ts - start_ts
        nms_time_avg = nms_time_avg / float(inferenced_pcs)
        model_inference_time_avg = model_inference_time_avg / float(inferenced_pcs)
        single_object_inference_time_avg = single_object_inference_time_avg / float(inferenced_pcs)
        ave_time = elapsed_time / float(inferenced_pcs)

        print("model\tnms\tinf\teps\tave")
        print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f"%(model_inference_time_avg, nms_time_avg, single_object_inference_time_avg, elapsed_time, ave_time))

