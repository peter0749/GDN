import os
import sys
import json
import warnings
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import deque

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
import importlib
import copy
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to configuration file (in JSON)")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)

    episodic_batch = deque(maxlen=config['batch_size'])

    if not os.path.exists(config['logdir']+'/ckpt'):
        os.makedirs(config['logdir']+'/ckpt')

    representation, dataset, _, base_model, model, optimizer, loss_function = import_model_by_setting(config)
    device = next(base_model.parameters()).device
    dataset.train()
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=True,
                            shuffle=True,
                            collate_fn=None)
    config = dataset.get_config()

    print('Num trainable params: %d'%count_parameters(base_model))

    iterations = config['iterations']
    start_ite = 1
    logger = SummaryWriter(config['logdir'])
    json.dump(config, open(config['logdir']+'/settings.json', "w"))

    best_tpr2 = -1.0

    if 'pretrain_path' in config and os.path.exists(config['pretrain_path']):
        states = torch.load(config['pretrain_path'])
        base_model.load_state_dict(states['base_model'])
        if 'loss_state' in states and (not states['loss_state'] is None) and hasattr(loss_function, 'load_state_dict'):
            loss_function.load_state_dict(states['loss_state'])
        best_tpr2 = states['best_tpr2']
        optimizer.load_state_dict(states['optimizer_state'])
        start_ite = states['ite'] + 1

    n_attempt = 0
    n_success = 0
    loss = 0.0

    model.train()
    dataset.train()
    batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
    for ite in range(1, iterations+1):
        try:
            pc, pc_clean, pc_origin = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
            pc, pc_clean, pc_origin = next(batch_iterator)
        with torch.no_grad():
            pc_cuda = torch.from_numpy(pc).cuda()
            pred, ind, att, l21 = model(pc_cuda)
            pc_subsampled = pointnet2_utils.gather_operation(pc_cuda.transpose(1, 2).contiguous(), ind)
            pc_npy = pc_subsampled.cpu().transpose(1, 2).numpy().astype(np.float32)
            pred = pred.cpu().numpy().astype(np.float32)
            pred_poses = np.asarray(decode_euler_feature(
                        pc_npy[0],
                        pred[0].reshape(1,-1),
                        *pred[0].shape[:-1],
                        config['hand_height'],
                        config['gripper_width'],
                        config['thickness_side'],
                        config['rot_th'],
                        config['trans_th'],
                        500, # max number of candidate
                        -np.inf, # threshold of candidate
                        100,  # max number of grasp in NMS
                        config['num_workers'],    # number of threads
                        True  # use NMS
            ), dtype=np.float32) # (?, 3, 4)
        successful_grasps = []
        with Pool(processes=args.workers) as pool:
            pool_results = []
            for n, pose in enumerate(pred_poses):
                pose_in_clean = np.copy(pose)
                pose_in_clean[:3, 3:4] += pc_origin.reshape(3, 1)
                gripper_outer1, gripper_outer2 = generate_gripper_edge(self.config['gripper_width']+self.config['thickness']*2, self.config['hand_height'], pose_in_clean, self.config['thickness_side'], backward=0.20)[1:]
                gripper_inner1, gripper_inner2 = generate_gripper_edge(self.config['gripper_width'], self.config['hand_height'], pose_in_clean, self.config['thickness_side'])[1:]
                outer_pts = crop_index(pc_clean, gripper_outer1, gripper_outer2)
                if len(outer_pts) == 0:
                    continue
                inner_pts = crop_index(pc_clean, gripper_inner1, gripper_inner2, search_idx=outer_pts)
                if len(outer_pts) - len(np.intersect1d(inner_pts, outer_pts)) > 3:
                    continue
                pool_results.append((n, inner_pts, pool.apply_async(eval_fn, (pc_clean[outer_pts], pose, config))))
            for n, (idx, r) in pool_results:
                if r.get():
                    successful_grasps.append((pred_poses[n], idx))
        if len(successful_grasps) > 0:
            episodic_batch.append((pc, successful_grasps)) # add to memory
        n_attempt += len(pred_poses)
        n_success += len(successful_grasps)
        print("iteration: %d | #attempt: %d | #success: %d | loss: %.2f"%(ite, n_attempt, n_success, loss))
        if len(episodic_batch) == config['batch_size']:
            feature_volume_batch = np.zeros((config['batch_size'], *config['output_dim']), dtype=np.float32) # TODO: can make it rolling
            pc_batch = []
            for b in range(config['batch_size']):
                for n, (pc, successful_grasps) in enumerate(episodic_batch[b]):
                    pc_batch.append(pc)
                    for pose, ind in successful_grasps:
                        (xyz,
                         roll,
                         pitch_index,
                         pitch_residual,
                         yaw_index,
                         yaw_residual) = representation.grasp_representation(pose,
                         pc,
                         ind)
                        representation.update_feature_volume(feature_volume_batch[b], ind, xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual)
            pc = torch.from_numpy(np.asarray(pc_batch, dtype=np.float32)).cuda()
            volume = torch.from_numpy(feature_volume_batch).cuda()
            optimizer.zero_grad()
            pred, ind, att, l21 = model(pc)
            l21 = l21.mean()
            (loss, foreground_loss, cls_loss,
                x_loss, y_loss, z_loss,
                rot_loss, ws, uncert) = loss_function(pred, ind, att, volume)
            loss += config['l21_reg_rate'] * l21 # l21 regularization (increase diversity)
            loss.backward()
            optimizer.step()

    logger.close()
