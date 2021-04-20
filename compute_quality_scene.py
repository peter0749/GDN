# coding: utf-8

import open3d # import first to avoid segfault
import sys
import os
import glob
import gc
import math
import numpy as np
import numba as nb
import pickle
import json
import warnings
from tqdm import tqdm
import matplotlib
import multiprocessing
from multiprocessing import Pool
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import pcl
    import struct
    import ctypes
    _VISUALIZE_USE_PCL = True
except ImportError:
    _VISUALIZE_USE_PCL = False
    print("Failed to import pcl. The point clouds will be display in grayscale.")


from gdn.utils import *

from argparse import ArgumentParser

warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)
warnings.filterwarnings('ignore', category=nb.NumbaPerformanceWarning)

def decode_pcl_rgb(f):
    s = struct.pack('>f', f)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value
    r = max(float((pack & 0x00FF0000)>> 16) / 255.0, 0)
    g = max(float((pack & 0x0000FF00)>> 8)  / 255.0, 0)
    b = max(float((pack & 0x000000FF))      / 255.0, 0)
    return (r,g,b)

def plot_gripper_tube(x1, x2, color = 'gray', tube_radius = 8.0):
    return go.Scatter3d(x = [x1[0], x2[0]], y = [x1[1], x2[1]], z = [x1[2], x2[2]],
            mode = 'lines',
            marker=dict(size=0.0,color=(0,0,0), opacity=0.0),
            line=dict(color=color, width=tube_radius))

def eval_fn(pc_npy, pose, rot_th, config, verbose):
    matched = reevaluate_antipodal_grasp(pc_npy, pose, max_degree=rot_th,
                hand_height = config['hand_height'],
                gripper_width = config['gripper_width'],
                thickness_side = config['thickness_side'],
                verbose=verbose)[0]
    return matched

def compute_match(ps, pc_id, predict_prefix='', pc_prefix='', rot_th=5.0, topK=10, vis_conf=False, vis_gamma=0.6, vis_proposal=False, use_vis=False):
    clip = topK * 2
    pc_npy = np.load(pc_prefix+'/'+pc_id+'.npy') # load corresponding prediction ordered by confidence
    z_min = pc_npy[...,2].min()
    pc_npy = pc_npy[pc_npy[...,2]>z_min+0.005]
    '''
    cloud = pcl.PointCloud(pc_npy)
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(*([0.0015,]*3))
    cloud_filtered = sor.filter()
    pc_npy = np.asarray(cloud_filtered, dtype=np.float32)
    del cloud, sor, cloud_filtered
    '''
    if use_vis:
        fig = mlab.figure(bgcolor=(0,0,0))
        col = (pc_npy[:,2] - pc_npy[:,2].min()) / (pc_npy[:,2].max() - pc_npy[:,2].min()) + 0.33
        mlab.points3d(pc_npy[:,0], pc_npy[:,1], pc_npy[:,2], col, scale_factor=0.0015, scale_mode='none', mode='sphere', colormap='plasma', opacity=1.0, figure=fig)
    results = []
    pred_poses_npy = np.load(predict_prefix+'/'+pc_id+'.npy')[:clip]

    with Pool(processes=args.workers) as pool:
        pool_results = []
        for n_cnt, (score, pc_id_m, row_n) in enumerate(ps):
            if n_cnt == clip:
                break
            assert pc_id  == pc_id_m
            pose = pred_poses_npy[row_n] # load corresponding prediction ordered by confidence
            gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width']+1e-5, config['hand_height'], pose, config['thickness_side'])
            enclosed_idx = crop_index(pc_npy, gripper_outer1, gripper_outer2)
            pool_results.append(pool.apply_async(eval_fn, (pc_npy[enclosed_idx], pose, rot_th, config, args.verbose)))
        results = [ r.get() for r in pool_results ]

    if use_vis:
        for n_cnt, (score, pc_id_m, row_n) in enumerate(ps):
            if n_cnt == clip:
                break
            pose = pred_poses_npy[row_n] # load corresponding prediction ordered by confidence
            matched = results[n_cnt]
            if use_vis and n_cnt<topK:
                gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
                gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
                center_bottom = (gripper_l+gripper_r) / 2.0
                approach = gripper_l_t-gripper_l
                approach = approach / np.linalg.norm(approach) # norm must > 0
                wrist_center = center_bottom - approach * 0.05

                if vis_conf:
                    control_p = np.clip((n_cnt/min(topK, len(ps)))**vis_gamma, 0, 1)
                    green_hsv = matplotlib.colors.rgb_to_hsv(np.array([0, 100, 0]))
                    red_hsv   = matplotlib.colors.rgb_to_hsv(np.array([255, 0, 0]))
                    s = np.asarray(np.clip(matplotlib.colors.hsv_to_rgb(control_p * red_hsv + (1.-control_p) * green_hsv), 0, 255), dtype=np.uint8)
                elif vis_proposal:
                    s = (100,100,100,178)
                else:
                    s = (0,100,0,178) if matched else (255,0,0,178)

                s = tuple(np.asarray(s[:3], dtype=np.float32) / 255.0)
                mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=0.003, color=s, opacity=0.8, figure=fig)
                mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=0.003, color=s, opacity=0.8, figure=fig)
                mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=0.003, color=s, opacity=0.8, figure=fig)
                mlab.plot3d([center_bottom[0], wrist_center[0]], [center_bottom[1], wrist_center[1]], [center_bottom[2], wrist_center[2]], tube_radius=0.003, color=s, opacity=0.8, figure=fig)
        mlab.savefig(figure=fig, filename='out.png')
        mlab.show()
        mlab.close(all=True)
    return results

def AP(results, topK=10):
    precisions = []
    tp = 0
    for n, matched in enumerate(results):
        if n >= topK:
            break
        if matched:
            tp += 1
            precisions.append(tp / (n + 1.))
    return np.mean(precisions) if tp>0 else 0

if __name__ == '__main__':
    # In[14]:
    thresholds = [30,]
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("pred", type=str, help="Prediction prefix")
    parser.add_argument("output", type=str, help="Output file in JSON format")
    parser.add_argument("--pc_path", type=str, required=True, help="Point cloud path for visualization")
    parser.add_argument("--top_K", type=int, default=10, help="Top-K for AP computation [10]")
    parser.add_argument("--workers", type=int, default=1, help="")
    parser.add_argument("--specify", type=str, default="", help="")
    parser.add_argument("--visualize_confidence", action="store_true", help="Visualize confidence of grasps")
    parser.add_argument("--visualize_proposal", action="store_true", help="")
    parser.add_argument("--visualize_gamma", type=float, default=1.0, help="")
    parser.add_argument("--use_vis", help="",
                    action="store_true")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = json.load(fp)

    if args.use_vis:
        try:
            from mayavi import mlab
        except ImportError:
            args.use_vis = False
            print("Failed to import mayavi. Visualization is disabled.")

    pred_prefix = args.pred
    suffix = '.npy'
    output = args.output
    pc_prefix = None
    pc_prefix = args.pc_path
    topK = args.top_K
    if len(args.specify)>0 and os.path.exists(args.specify):
        pred_files = [args.specify]
    else:
        pred_files = glob.glob(pred_prefix + '/*.meta')
    np.random.shuffle(pred_files)
    APs_at_threshold = {}
    pbar = tqdm(total=len(pred_files))
    for path in pred_files:
        obj = path.split('/')[-2]
        pc_id = path.split('/')[-1][:-5]
        with open(path, 'rb') as fp:
            meta = list(filter(lambda x: x[0]!=-np.inf, pickle.load(fp)))
            meta.sort(reverse=True) # sort by confidence score
        aps_disp = ''
        for rot_th in thresholds:
            results = compute_match(meta, pc_id, rot_th=rot_th, pc_prefix=pc_prefix, predict_prefix=pred_prefix, topK=topK, vis_conf=args.visualize_confidence, vis_gamma=args.visualize_gamma, vis_proposal=args.visualize_proposal, use_vis=args.use_vis)
            ap = AP(results, topK=topK)
            aps_disp = aps_disp + 'AP@%d: %.2f | '%(rot_th, ap)
            if not rot_th in APs_at_threshold:
                APs_at_threshold[rot_th] = []
            APs_at_threshold[rot_th].append(ap)
        pbar.set_description('%s%s | %s'%(aps_disp, obj, pc_id))
        pbar.update(1)
    pbar.close()
    del pbar
    mAP_at_threshold = {}
    for rot_th in thresholds:
        mAP_at_threshold[rot_th] = np.mean(APs_at_threshold[rot_th])
    for rot_th in thresholds:
        print('mAP@%d: %.4f'%(rot_th, mAP_at_threshold[rot_th]))
    with open(output, 'w') as fp:
        fp.write(json.dumps({'mAP': mAP_at_threshold, 'APs': APs_at_threshold}))
