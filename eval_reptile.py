# coding: utf-8

#from mayavi import mlab
#mlab.options.offscreen = True

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

from pointnet2.utils import pointnet2_utils

from nms import decode_euler_feature
from nms import initEigen, sanity_check
initEigen(0)

from gdn.utils import *
from gdn.detector.utils import *
from gdn import import_model_by_setting
import importlib
import copy
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to configuration file (in JSON)")
    parser.add_argument("task_data", type=str, help="")
    parser.add_argument("task_label", type=str, help="")
    parser.add_argument("weight_path", type=str, help="Path to the trained model weights")
    parser.add_argument("ntrain", type=int, help="")
    parser.add_argument("nepoch", type=int, help="")
    parser.add_argument("batch_size", type=int, help="")
    parser.add_argument("output_dir", type=str, help="Path to the results")
    parser.add_argument("--naug", type=int, default=0, help="")
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
        #config['input_points'] = 20000 # TODO: DELETE ME

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    representation, dataset, my_collate_fn, base_model, model, optimizer, loss_function = import_model_by_setting(config, mode='val')
    assert args.ntrain < len(dataset)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=False,
                            shuffle=True,
                            collate_fn=my_collate_fn)
    print('Num params: %d'%count_parameters(base_model))

    states = torch.load(args.weight_path)
    base_model.load_state_dict(states['base_model'])
    if 'loss_state' in states and (not states['loss_state'] is None) and hasattr(loss_function, 'load_state_dict'):
        loss_function.load_state_dict(states['loss_state'])

    batch_iterator = iter(dataloader)

    # Prepare batches
    if args.ntrain > 0:
        print("Preparing training batches...")
        if args.naug <= 0:
            Xs = [] # point clouds
            Ys = np.zeros((args.ntrain,
                           config['input_points'],
                           *config['output_dim']),
                          dtype=np.float32) # grasp volumes
            for b in tqdm(range(args.ntrain)):
                batch = next(batch_iterator)
                pc, hand_poses = batch[:2]
                pc_npy = pc[0].numpy().astype(np.float32)
                Xs.append(pc)
                for pose in reversed(hand_poses[0]): # overwrite lower antipodal grasps
                    gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
                    enclosed_pts = crop_index(pc_npy, gripper_outer1, gripper_outer2)
                    if len(enclosed_pts)==0:
                        continue
                    (xyz,
                     roll,
                     pitch_index,
                     pitch_residual,
                     yaw_index,
                     yaw_residual) = representation.grasp_representation(pose,
                     pc_npy, # shape: (N, 3)
                     enclosed_pts)
                    representation.update_feature_volume(Ys[b], enclosed_pts, xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual)
            Xs = torch.cat(Xs).float()
            Ys = torch.from_numpy(Ys).float()
        else:
            Xs = [] # point clouds
            Ys = np.zeros((args.ntrain * args.naug,
                           config['input_points'],
                           *config['output_dim']),
                          dtype=np.float32) # grasp volumes
            for b in tqdm(range(args.ntrain)):
                batch_original = next(batch_iterator)
                for n, rz in enumerate(np.linspace(0, 2.0*np.pi, args.naug+1)[:-1]):
                    pc, hand_poses = copy.deepcopy(batch_original)[:2]
                    pc_npy = pc[0].numpy().astype(np.float32) # (N, 3)
                    hand_poses = np.asarray(hand_poses[0], dtype=np.float32) # (M, 3, 4)

                    if n > 0:
                        # Apply rotation
                        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                                       [np.sin(rz), np.cos(rz), 0],
                                       [0, 0, 1]], dtype=np.float32)
                        RzT = Rz.T
                        pc_npy = np.ascontiguousarray(pc_npy@RzT) #(Rz@pc_npy.T).T
                        t = hand_poses[:,:3,3]@RzT #(Rz@hand_poses[:,:3,3].T).T # (M, 3)
                        vx = hand_poses[:,:3,0]@RzT #(Rz@hand_poses[:,:3,0].T).T # (M, 3)
                        vy = hand_poses[:,:3,1]@RzT #(Rz@hand_poses[:,:3,1].T).T # (M, 3)
                        vz = hand_poses[:,:3,2]@RzT #(Rz@hand_poses[:,:3,2].T).T # (M, 3)
                        hand_poses = np.ascontiguousarray(np.transpose(np.stack((vx, vy, vz, t)), (1, 2, 0))) # (4, M, 3) -> (M, 3, 4)

                    '''
                    fig = mlab.figure(bgcolor=(0,0,0))
                    col = (pc_npy[:,2] - pc_npy[:,2].min()) / (pc_npy[:,2].max() - pc_npy[:,2].min()) + 0.33
                    mlab.points3d(pc_npy[:,0], pc_npy[:,1], pc_npy[:,2], col, scale_factor=0.0015, scale_mode='none', mode='sphere', colormap='plasma', opacity=1.0, figure=fig)
                    '''

                    Xs.append(pc)
                    #for pose in reversed(hand_poses[:10]): # overwrite lower antipodal grasps
                    for pose in reversed(hand_poses): # overwrite lower antipodal grasps
                        gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
                        '''
                        gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
                        center_bottom = (gripper_l+gripper_r) / 2.0
                        approach = gripper_l_t-gripper_l
                        approach = approach / np.linalg.norm(approach) # norm must > 0
                        wrist_center = center_bottom - approach * 0.05
                        mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=0.003, color=(0, 1, 0), opacity=0.8, figure=fig)
                        mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=0.003, color=(0, 1, 0), opacity=0.8, figure=fig)
                        mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=0.003, color=(0, 1, 0), opacity=0.8, figure=fig)
                        mlab.plot3d([center_bottom[0], wrist_center[0]], [center_bottom[1], wrist_center[1]], [center_bottom[2], wrist_center[2]], tube_radius=0.003, color=(0, 1, 0), opacity=0.8, figure=fig)
                        '''
                        enclosed_pts = crop_index(pc_npy, gripper_outer1, gripper_outer2)
                        if len(enclosed_pts)==0:
                            continue
                        (xyz,
                         roll,
                         pitch_index,
                         pitch_residual,
                         yaw_index,
                         yaw_residual) = representation.grasp_representation(pose,
                         pc_npy, # shape: (N, 3)
                         enclosed_pts)
                        representation.update_feature_volume(Ys[b*args.naug+n], enclosed_pts, xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual)
                    #mlab.savefig('example-%d.png'%n, figure=fig)
                    #mlab.clf()
            Xs = torch.cat(Xs).float()
            Ys = torch.from_numpy(Ys).float()

        # Start fine-tune
        print("Fine-tuning...")
        model.train()
        for _ in tqdm(range(args.nepoch)):
            batch_inds = np.random.permutation(len(Xs))
            for start in range(0, len(Xs), args.batch_size):
                mbinds = batch_inds[start:start+args.batch_size]
                X = Xs[mbinds].cuda()
                Y = Ys[mbinds].cuda()
                optimizer.zero_grad()
                pred, ind, att = model(X)[:3]
                (loss, foreground_loss, cls_loss,
                    x_loss, y_loss, z_loss,
                    rot_loss, ws, uncert) = loss_function(pred, ind, att, Y)
                loss.backward()
                optimizer.step()

    # Start evaluate
    model.eval()
    with torch.no_grad():
        while True:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break
            pc, gt, pc_origin, scene_ids = batch
            pc_cuda = pc.cuda()
            pred, ind, att = model(pc_cuda)[:3]
            pc_subsampled = pointnet2_utils.gather_operation(pc_cuda.transpose(1, 2).contiguous(), ind)
            pc_npy = pc_subsampled.cpu().transpose(1, 2).numpy() + pc_origin # (B, N, 3) + (B, 1, 3)
            pc_npy = pc_npy.astype(np.float32)
            pred = pred.cpu().numpy().astype(np.float32)
            pred_poses = [
                    np.asarray(sanity_check(pc_npy[b],
                        np.asarray(decode_euler_feature(
                        pc_npy[b],
                        pred[b].reshape(1,-1),
                        *pred[b].shape[:-1],
                        config['hand_height'],
                        config['gripper_width'],
                        config['thickness_side'],
                        config['rot_th'],
                        config['trans_th'],
                        300, # max number of candidate
                        -np.inf, # threshold of candidate
                        60,  # max number of grasp in NMS
                        config['num_workers'],    # number of threads
                        True  # use NMS
                        ), dtype=np.float32)
                        , 2,
                        config['gripper_width'],
                        config['thickness'],
                        config['hand_height'],
                        config['thickness_side'],
                        config['num_workers'] # num threads
                    ), dtype=np.float32) for b in range(len(pc_npy)) ]
            #pred_poses = representation.retrive_from_feature_volume_batch(pc_npy, pred.cpu().numpy(), n_output=300, threshold=-1e8, nms=True)
            #pred_poses = representation.filter_out_invalid_grasp_batch(pc_npy, pred_poses)
            for pose, id_ in zip(pred_poses, scene_ids):
                prefix = args.output_dir
                meta = [(-n, id_, n) for n in range(len(pose))]
                with open(prefix+'/'+id_+'.meta', 'wb') as fp:
                    pickle.dump(meta, fp, protocol=2, fix_imports=True)
                np.save(prefix+'/'+id_+'.npy', pose)
                print('Processed: {} {:s}'.format(id_, str(pose.shape)))
