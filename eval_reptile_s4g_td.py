# coding: utf-8

import open3d

#from mayavi import mlab
#mlab.options.offscreen = True

import os
import time
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
from sklearn.cluster import KMeans
from dppy.finite_dpps import FiniteDPP

from nms import decode_euler_feature
from nms import initEigen, sanity_check
initEigen(0)

from gdn.utils import *
from gdn.detector.utils import *
from gdn import import_model_by_setting
from gdn.representation.s4g_focal_maml.activation import cvtD6SO3
import importlib
import copy
from argparse import ArgumentParser

@nb.njit
def rotDist(x, y):
    return np.arcsin(np.linalg.norm(np.eye(3) - np.dot(x[:3,:3], y[:3,:3].T), ord=None ) / (2*np.sqrt(2)))

@nb.njit
def makeAffinityMatrix(poses, sigma_r=0.5, sigma_t=0.25):
    A = np.eye(len(poses), dtype=np.float32)
    for i in range(len(poses)):
        for j in range(i+1, len(poses)):
            d = rotDist(poses[i,:3,:3], poses[j,:3,:3])
            s_r = np.exp(-d / ((2. * sigma_r) ** 2))
            if not np.isfinite(s_r):
                s_r = 0.0
            d = np.linalg.norm(poses[i,:3,3]-poses[j,:3,3], ord=2)
            s_t = np.exp(-d / ((2. * sigma_t) ** 2))
            A[i,j] = A[j,i] = max(s_r * s_t, 1e-8)
    return A

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
    parser.add_argument("--nrot", type=int, default=5, help="Simulate number of annotation labeled by humans.")
    parser.add_argument("--npos", type=int, default=7, help="Simulate number of annotation labeled by humans.")
    parser.add_argument("--ngrasp", type=int, default=10, help="Simulate number of annotation labeled by humans.")
    parser.add_argument("--max_eval", type=int, default=60, help="")
    parser.add_argument("--pu_loss", action="store_true", default=False)
    parser.add_argument("--pu_loss_type", type=str, default="sigmoid", help="")
    parser.add_argument("--nneighbors", type=int, default=3, help="")
    parser.add_argument("--nnode", type=int, default=3000, help="")
    parser.add_argument("--alpha", type=float, default=0.99, help="")
    parser.add_argument("--conf_lo", type=float, default=0.2, help="")
    #parser.add_argument("--conf_hi", type=float, default=0.3, help="")
    #parser.add_argument("--conf_bw", type=float, default=0.8, help="")
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
        config['return_embedding'] = True

    if 'pu_loss' in config:
        del config['pu_loss']
    if args.pu_loss:
        config['pu_loss'] = args.pu_loss
        config['pu_loss_type'] = args.pu_loss_type

    k_dpp = args.nrot
    n_clusters = args.npos
    mcmc_trials = 5
    mcmc_iters = 300

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
    print('Num params: %d'%count_parameters(base_model), flush=True)

    states = torch.load(args.weight_path)
    base_model.load_state_dict(states['base_model'])
    if 'loss_state' in states and (not states['loss_state'] is None) and hasattr(loss_function, 'load_state_dict'):
        loss_function.load_state_dict(states['loss_state'])

    batch_iterator = iter(dataloader)

    if args.ntrain > 0:
        # Prepare batches
        print("Preparing GT...", flush=True)
        Xs = [] # point clouds
        Ys = np.zeros((args.ntrain,
                       config['input_points'],
                       *config['output_dim']),
                      dtype=np.float32)
        for b in tqdm(range(args.ntrain)):
            batch = next(batch_iterator)
            pc, hand_poses, pc_origin, scene_ids = batch
            poses = hand_poses[0]
            if len(poses) > n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, max_iter=100, n_init=3, verbose=False)
                kmeans.fit(poses[...,3])
                selected_grasps = []
                for l in range(n_clusters):
                    poses_l = poses[kmeans.labels_==l]
                    if len(poses_l) > k_dpp:
                        A = makeAffinityMatrix(poses_l)
                        DPP = FiniteDPP('likelihood', **{'L': A})
                        best_samples = []
                        best_scores = np.inf
                        for _ in range(mcmc_trials):
                            samples = DPP.sample_mcmc_k_dpp(size=k_dpp, nb_iter=mcmc_iters)
                            sim = 0.0
                            for i in range(k_dpp):
                                for j in range(i+1, k_dpp):
                                    sim += A[samples[i],samples[j]]
                            if sim < best_scores:
                                best_scores = sim
                                best_samples = samples
                        selected_grasps += [ poses_l[i] for i in best_samples ]
                    else:
                        selected_grasps += list(poses_l)
            else:
                selected_grasps = poses
            selected_grasps = np.asarray(selected_grasps, dtype=np.float32)
            if len(selected_grasps) > args.ngrasp:
                selected_grasps = selected_grasps[ np.random.choice(len(selected_grasps), args.ngrasp, replace=False)  ]
            pc_npy = pc[0].numpy().astype(np.float32)
            Xs.append(pc)
            for pose in selected_grasps:
                gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
                enclosed_pts = crop_index(pc_npy, gripper_outer1, gripper_outer2)
                if len(enclosed_pts)==0:
                    continue
                (xyz, d9_rot) = representation.grasp_representation(pose,
                     pc_npy, # shape: (N, 3)
                     enclosed_pts)
                representation.update_feature_volume(Ys[b], enclosed_pts, xyz, d9_rot)
        Xs = torch.cat(Xs).float()
        Ys = torch.from_numpy(Ys).float()
        Ys[...,0].clamp_(0, 1)
        print("Positive / batch (before): %.4f"%(torch.sum(Ys[...,0]>0) / Ys.size(0)), flush=True)

        for t_epoch in range(1, args.nepoch+1):
            print("=== Epoch #%d ==="%t_epoch)
            Hs = []
            Hs_unk = []
            Inds_sub = []

            # Start fine-tune
            print("Generating pseudo-labels...", flush=True)
            model.train() # simulate fine-tune stage

            with torch.no_grad():
                for start in tqdm(range(0, len(Xs), args.batch_size)):
                    X = Xs[start:start+args.batch_size].cuda()
                    h = model(X)[1]
                    ind_sub = pointnet2_utils.furthest_point_sample(X, args.nnode) # (B, K)
                    h_unk = pointnet2_utils.gather_operation(h, ind_sub) # (B, 128, nnode)
                    Hs.append(h.cpu())
                    Hs_unk.append(h_unk.cpu())
                    Inds_sub.append(ind_sub.cpu().long())
            Hs = torch.cat(Hs, 0)
            Hs_unk = torch.cat(Hs_unk, 0)
            Inds_sub = torch.cat(Inds_sub, 0)
            C = Hs.size(1)

            time_n = 0
            time_s = 0

            with torch.no_grad():
                Ys_new = torch.zeros_like(Ys)
                gt_mask = tuple((Ys[...,0]>0).nonzero().transpose(0, 1).contiguous())
                Ys_shape = Ys.size()
                Hs_pos = Hs.transpose(1, 2)[gt_mask].view(-1, C).transpose(0, 1).contiguous()
                Ys_pos = Ys.view(Ys.size(0), Ys.size(1), -1)[gt_mask].contiguous()
                # Label propagation
                for b in range(Xs.size(0)):
                    s_time = time.time()
                    Wb = make_propagation_kernel(torch.cat((Hs_unk[b], Hs_pos), -1), n_neighbors=args.nneighbors, alpha=args.alpha)
                    Yb = torch.zeros(args.nnode, Ys_pos.size(1), dtype=Ys_pos.dtype, device=Ys_pos.device) # unk: (nnode, -1)
                    Yb = torch.cat((Yb, Ys_pos), 0) # + has label (nnode+|Y|, -1)
                    Z = (1-args.alpha) * torch.mm(Wb, Yb).reshape(Yb.size(0), *Ys_shape[2:]) # (K, K) @ (K, ?) -> (K, ?)
                    Z = Z[:args.nnode]
                    print("C_min C_max: %.4e %.4e"%(Z[...,0].min(), Z[...,0].max()))

                    # Convert SO3 to D6
                    d6 = Z[...,1:10].view(-1, 3, 3)[:,:,:2].reshape(-1, 6)
                    Z[...,1:10] = cvtD6SO3(d6).reshape(Z.size(0), 9)

                    # ignore pseudo-labels w/ low conf.
                    ignore = Z[...,0] < args.conf_lo
                    #Z[...,0] = ((Z[...,0] - args.conf_lo) / (args.conf_hi - args.conf_lo)).clamp(0, 1) * args.conf_bw + (1.0 - args.conf_bw) / 2.0
                    Z[...,0] = 1.0
                    Z[ignore,0] = 0.0
                    Ys_new[b][Inds_sub[b]] = Z
                    time_s += time.time() - s_time
                    time_n += 1.
                Ys_new[gt_mask] = Ys[gt_mask]
            print("Avg. LP time: %.2f"%(time_s/time_n))
            print("Positive / batch (after): %.4f"%(torch.sum(Ys_new[...,0]>0) / Ys_new.size(0)))
            print("Computing collision...")
            pred_poses = representation.retrive_from_feature_volume_batch(Xs.numpy().astype(np.float32), Ys_new.numpy().astype(np.float32), n_output=10000, threshold=0.0, nms=False)
            # Dirty hack...
            config_bak = copy.deepcopy(representation.config)
            representation.config['hand_height'] = config['hand_height_eval']
            representation.config['gripper_width'] = config['gripper_width_eval']
            representation.config['thickness_side'] = config['thickness_side_eval']
            representation.config['thickness'] = config['thickness_eval']
            pred_poses = representation.filter_out_invalid_grasp_batch(Xs.numpy().astype(np.float32), pred_poses)
            representation.config = config_bak
            print("Generating new Ys...")
            Ys_new = Ys_new.numpy().astype(np.float32)
            for b in range(Xs.size(0)):
                pc_npy = Xs[b].numpy().astype(np.float32)
                for score, pose in reversed(pred_poses[b]):
                    gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
                    enclosed_pts = crop_index(pc_npy, gripper_outer1, gripper_outer2)
                    if len(enclosed_pts)==0:
                        continue
                    (xyz, d9_rot) = representation.grasp_representation(pose,
                         pc_npy, # shape: (N, 3)
                         enclosed_pts)
                    representation.update_feature_volume(Ys_new[b], enclosed_pts, xyz, d9_rot)
                    Ys_new[b,enclosed_pts,0] = score
            Ys_new = torch.from_numpy(Ys_new)
            # Start fine-tune
            print("Fine-tuning...", flush=True)
            model.train()
            batch_inds = np.random.permutation(len(Xs))
            for start in range(0, len(Xs), args.batch_size):
                mbinds = batch_inds[start:start+args.batch_size]
                X = Xs[mbinds].cuda()
                Y_new = Ys_new[mbinds].cuda() # (B, N, 16, 17, 8)
                optimizer.zero_grad()
                pred, h = model(X)
                loss, cls_loss = loss_function(pred, Y_new)[:2]
                print("cls_loss: %.2f"%cls_loss, flush=True)
                loss.backward()
                optimizer.step()

    # Start evaluate
    model.eval()
    with torch.no_grad():
        n_evaled = 0
        while n_evaled < args.max_eval:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                break
            pc, gt, pc_origin, scene_ids = batch
            pc_cuda = pc.cuda()
            pred = model(pc_cuda)[0]
            pc_npy = pc.cpu().numpy() + pc_origin # (B, N, 3) + (B, 1, 3)
            pc_npy = pc_npy.astype(np.float32)
            pred = pred.cpu().numpy().astype(np.float32)
            pred_poses = representation.retrive_from_feature_volume_batch(pc_npy, pred, n_output=2000, threshold=-1e8, nms=True)
            pred_poses = representation.filter_out_invalid_grasp_batch(pc_npy, pred_poses)
            for pose, id_ in zip(pred_poses, scene_ids):
                prefix = args.output_dir
                if len(pose)>0:
                    lscore, lpose_3x4 = zip(*pose)
                    lpose_3x4 = np.stack(lpose_3x4)
                else:
                    lscore, lpose_3x4 = [], np.empty((0, 3, 4), dtype=np.float32)
                lpose_3x4 = lpose_3x4.astype(np.float32)
                meta = [(lscore[n], id_, n) for n in range(len(pose))]
                with open(prefix+'/'+id_+'.meta', 'wb') as fp:
                    pickle.dump(meta, fp, protocol=2, fix_imports=True)
                np.save(prefix+'/'+id_+'.npy', lpose_3x4)
                print('Processed: {} {:s}'.format(id_, str(lpose_3x4.shape)))
                n_evaled += 1
