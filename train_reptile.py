import os
import sys
import json
import warnings
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import deque
from multiprocessing import Pool

import numba as nb
warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from nms import decode_euler_feature, initEigen
initEigen(0)

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

def eval_fn(pc_npy, pose, angle, config):
    matched = reevaluate_antipodal_grasp(pc_npy, pose, max_degree=angle,
                hand_height = config['hand_height'],
                gripper_width = config['gripper_width'],
                thickness_side = config['thickness_side'],
                verbose=False)[0]
    return matched

if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)

    memory_bank = deque(maxlen=config['memory_bank_size'])

    if not os.path.exists(config['logdir']+'/ckpt'):
        os.makedirs(config['logdir']+'/ckpt')

    representation, dataset, _, base_model, __, optimizer, loss_function = import_model_by_setting(config)
    model = base_model
    device = next(base_model.parameters()).device
    dataset.train()
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=False,
                            shuffle=True,
                            collate_fn=None)
    config = dataset.get_config()

    print('Num trainable params: %d'%count_parameters(base_model))

    iterations = config['iterations']
    start_ite = 1
    logger = SummaryWriter(config['logdir'])
    json.dump(config, open(config['logdir']+'/settings.json', "w"))

    n_attempt = 0
    n_success = 0
    loss_mean = 0.0
    temperature = config['annealing_max']

    if 'pretrain_path' in config and os.path.exists(config['pretrain_path']):
        states = torch.load(config['pretrain_path'])
        base_model.load_state_dict(states['base_model'])
        if 'loss_state' in states and (not states['loss_state'] is None) and hasattr(loss_function, 'load_state_dict'):
            loss_function.load_state_dict(states['loss_state'])
        optimizer.load_state_dict(states['optimizer_state'])
        if 'ite' in states:
            start_ite = states['ite'] + 1
        if 'n_attempt' in states:
            n_attempt = states['n_attempt']
        if 'n_success' in states:
            n_success = states['n_success']
        if 'temperature' in states:
            temperature = states['temperature']

    model.train()
    dataset.train()
    batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
    for ite in range(start_ite, iterations+1):
        try:
            pc, pc_clean = next(batch_iterator)[:2]
        except StopIteration:
            batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
            pc, pc_clean = next(batch_iterator)[:2]
        pc = pc.numpy().astype(np.float32)
        pc_clean = pc_clean.numpy().astype(np.float32)
        with torch.no_grad():
            pc_cuda = torch.from_numpy(pc).cuda()
            pred, ind, att = model.sampling(pc_cuda, temperature=temperature)
            temperature = max(config['annealing_min'], temperature * config['annealing_t'])
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
                        100, # max number of candidate
                        -np.inf, # threshold of candidate
                        100,  # max number of grasp in NMS
                        config['num_workers'],    # number of threads
                        False  # use NMS
            ), dtype=np.float32) # (?, 3, 4)
        for antipodal_angle in [15, 30, 45]:
            print("Testing with antipodal angle: %d"%antipodal_angle)
            sampled_grasps = []
            cases = np.zeros(3, dtype=np.float32)
            with Pool(processes=config['num_workers']) as pool:
                pool_results = []
                for n, pose in enumerate(pred_poses):
                    gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width']+config['thickness']*2, config['hand_height'], pose, config['thickness_side'], backward=0.20)[1:]
                    gripper_inner1, gripper_inner2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])[1:]
                    outer_pts = crop_index(pc_clean[0], gripper_outer1, gripper_outer2)
                    if len(outer_pts) == 0:
                        cases[2] += 1
                        continue
                    inner_pts = crop_index(pc_clean[0], gripper_inner1, gripper_inner2, search_idx=outer_pts)
                    if len(outer_pts) - len(np.intersect1d(inner_pts, outer_pts)) > 3:
                        cases[1] += 1
                        continue
                    gripper_inner1, gripper_inner2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])[1:]
                    inner_pts = crop_index(pc[0], gripper_inner1, gripper_inner2)
                    pool_results.append((n, inner_pts, pool.apply_async(eval_fn, (pc_clean[0,outer_pts], pose, antipodal_angle, config))))
                    cases[0] += 1
                for n, idx, r in pool_results:
                    if r.get():
                        sampled_grasps.append((pred_poses[n], idx))
            cases = cases / cases.sum()
            print('viable / collision / empty: ', cases)
            if cases[0] == 0:
                print('No viable grasps.')
                antipodal_angle = 180
                break
            if len(sampled_grasps) > 0:
                memory_bank.append((pc[0], sampled_grasps)) # add to memory
                break
        n_attempt += len(pred_poses)
        if antipodal_angle < 31:
            n_success += len(sampled_grasps)
        print("iteration: %d | #attempt: %d | #success: %d | antipodal_angle: %d | loss: %.2f | temperature: %.4f | memory_bank: %d / %d (%d)"%(ite, n_attempt, n_success, antipodal_angle, loss_mean, temperature, len(memory_bank), config['memory_bank_size'], config['task_size']))
        logger.add_scalar('n_attempt_vs_n_success_r', n_success / n_attempt, n_attempt)
        logger.add_scalar('n_attempt_vs_n_success', n_success, n_attempt)
        logger.add_scalar('n_iter_vs_n_success_r', n_success / ite, ite)
        logger.add_scalar('n_iter_vs_n_success', n_success, ite)
        logger.add_scalar('temperature', temperature, ite)
        if len(memory_bank) >= config['task_size']:
            feature_volume_batch = np.zeros((config['task_size'], config['input_points'], *config['output_dim']), dtype=np.float32) # TODO: can make it rolling
            pc_batch = []
            for b, memory_i in enumerate(np.random.choice(len(memory_bank), config['task_size'], replace=False)):
                pc, successful_grasps = memory_bank[memory_i]
                pc_batch.append(pc)
                for pose, ind in successful_grasps:
                    if len(ind) == 0:
                        continue
                    (xyz,
                     roll,
                     pitch_index,
                     pitch_residual,
                     yaw_index,
                     yaw_residual) = representation.grasp_representation(pose,
                     pc,
                     ind)
                    representation.update_feature_volume(feature_volume_batch[b], ind, xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual)
            # Train on this task
            weights_before = copy.deepcopy(base_model.state_dict())
            pc = torch.from_numpy(np.stack(pc_batch).astype(np.float32)).cuda()
            volume = torch.from_numpy(feature_volume_batch).cuda()
            dpp_sum = 0.0
            loss_sum = 0.0
            step_sum = 0.0
            for _ in range(config['innerepochs']):
                batch_inds = np.random.permutation(config['task_size'])
                for start in range(0, config['task_size'], config['batch_size']):
                    mbinds = batch_inds[start:start+config['batch_size']]
                    optimizer.zero_grad()
                    pred, ind, att, l21 = model(pc[mbinds])
                    l21 = l21.mean()
                    (loss, foreground_loss, cls_loss,
                        x_loss, y_loss, z_loss,
                        rot_loss, ws, uncert) = loss_function(pred, ind, att, volume[mbinds])
                    loss += config['l21_reg_rate'] * l21 # l21 regularization (increase diversity)
                    loss.backward()
                    optimizer.step()
                    dpp_sum += l21.item()
                    loss_sum += loss.item()
                    step_sum += 1.0
            weights_after = base_model.state_dict()
            outerstepsize = config['outerstepsize0'] * (1.0 - (ite-1.0) / iterations) # linear schedule
            base_model.load_state_dict({name : weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
                                        for name in weights_before})
            torch.save({
                    'base_model': base_model.state_dict(),
                    'loss_state': loss_function.state_dict() if hasattr(loss_function, 'state_dict') else None,
                    'optimizer_state': optimizer.state_dict(),
                    'ite': ite,
                    'n_attempt': n_attempt,
                    'n_success': n_success,
                    'temperature': temperature,
                    }, config['logdir'] + '/last.ckpt')
            dpp_mean = dpp_sum/step_sum
            loss_mean = loss_sum/step_sum
            logger.add_scalar('dpp', dpp_mean, ite)
            logger.add_scalar('loss', loss_mean, ite)

    logger.close()
