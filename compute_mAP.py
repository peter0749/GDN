# coding: utf-8

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
from pykdtree.kdtree import KDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.utils import *

from argparse import ArgumentParser

warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)
warnings.filterwarnings('ignore', category=nb.NumbaPerformanceWarning)

config = dict(
    hand_height = 0.11, # 11cm (x range)
    gripper_width = 0.08, # 8cm (y range)
    thickness_side = 0.01, # (z range)
    thickness = 0.01, # gripper_width + thickness*2 = hand_outer_diameter (0.08+0.01*2 = 0.1)
)

def compute_match(ps, ts, obj_id, pc_id, predict_prefix='', pc_prefix=None, rot_th=5.0, trans_th=0.02, topK=10):
    if not pc_prefix is None:
        pc_npy = np.load(pc_prefix+'/'+obj_id+'/clouds/'+pc_id+'.npy') # load corresponding prediction ordered by confidence
        if len(pc_npy)>2048:
            pc_npy = pc_npy[np.random.choice(len(pc_npy), 2048, replace=False)]
        fig = mlab.figure(bgcolor=(0,0,0))
        color_scale = np.copy(pc_npy[:,2])
        color_scale = (color_scale-color_scale.min()) / (color_scale.max()-color_scale.min())
        mlab.points3d(pc_npy[:,0], pc_npy[:,1], pc_npy[:,2], color_scale, scale_factor=0.004, scale_mode='none', mode='sphere', colormap='jet', opacity=1.0, figure=fig)
    ts_kd_tree = {k: KDTree(np.asarray(v)[:,:,3]) for (k, v) in ts.items() } # KD-tree for each object
    ts_matched = set() # ground truth pose set that matched
    results = []
    for n_cnt, (score, obj_id_m, pc_id_m, row_n) in enumerate(ps):
        assert obj_id == obj_id_m
        assert pc_id  == pc_id_m
        pose = np.load(predict_prefix+'/'+obj_id+'/'+pc_id+'.npy')[row_n] # load corresponding prediction ordered by confidence

        trans_matched = ts_kd_tree[obj_id].query(pose[:,3].reshape(1, 3), k=len(ts[obj_id]), distance_upper_bound=trans_th)[1][0]
        matched = False
        for i in trans_matched[trans_matched<len(ts[obj_id])]:
            if rot_match(pose, ts[obj_id][i], th=rot_th) and not (obj_id, pc_id, i) in ts_matched:
                matched = True
                ts_matched.add((obj_id, pc_id, i))
                break
        results.append(matched)
        if not pc_prefix is None and n_cnt<topK:
            gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
            gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
            center_bottom = (gripper_l+gripper_r) / 2.0
            approach = gripper_l_t-gripper_l
            approach = approach / np.linalg.norm(approach) # norm must > 0
            wrist_center = center_bottom - approach * 0.05

            mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=0.001, color=(0,1,0) if matched else (1,0,0), opacity=0.8, figure=fig)
            mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=0.001, color=(0,1,0) if matched else (1,0,0), opacity=0.8, figure=fig)
            mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=0.001, color=(0,1,0) if matched else (1,0,0), opacity=0.8, figure=fig)
            mlab.plot3d([center_bottom[0], wrist_center[0]], [center_bottom[1], wrist_center[1]], [center_bottom[2], wrist_center[2]], tube_radius=0.001, color=(0,1,0) if matched else (1,0,0), opacity=0.8, figure=fig)
    if not pc_prefix is None:
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
    thresholds = [5, 10, 15]
    parser = ArgumentParser()
    parser.add_argument("gt", type=str, help="Ground truth prefix")
    parser.add_argument("pred", type=str, help="Prediction prefix")
    parser.add_argument("gt_suffix", type=str, help="Suffix of the GT [_nms.npy]")
    parser.add_argument("output", type=str, help="Output file in JSON format")
    parser.add_argument("--pc_path", type=str, default="", help="Point cloud path for visualization")
    parser.add_argument("--top_K", type=int, default=10, help="Top-K for AP computation [10]")
    parser.add_argument("--specify", type=str, default="", help="")
    args = parser.parse_args()

    gt_prefix = args.gt
    pred_prefix = args.pred
    suffix = args.gt_suffix
    output = args.output
    pc_prefix = None
    if os.path.exists(args.pc_path):
        from mayavi import mlab
        pc_prefix = args.pc_path
    topK = args.top_K
    gt_id = os.listdir(gt_prefix)
    if len(args.specify)>0 and os.path.exists(args.specify):
        pred_files = [args.specify]
    else:
        pred_files = glob.glob(pred_prefix + '/*/*.meta')
    np.random.shuffle(pred_files)
    pred_id = set( [ x.split('/')[-2] for x in pred_files ] )
    obj2grasp = {}
    for obj_name in gt_id:
        #XXX_nms.npy
        obj_id = obj_name[:-len(suffix)]
        if not obj_id in pred_id: # train set or unknown
            continue
        path = gt_prefix + '/' + obj_name
        obj2grasp[obj_id] = np.load(path)
    APs_for_obj = {}
    APs_at_threshold = {}
    pbar = tqdm(total=len(pred_files))
    for path in pred_files:
        obj = path.split('/')[-2]
        pc_id = path.split('/')[-1][:-5]
        if not obj in APs_for_obj:
            APs_for_obj[obj] = {}
        with open(path, 'rb') as fp:
            meta = list(filter(lambda x: x[0]!=-np.inf, pickle.load(fp)))
            meta.sort(reverse=True) # sort by confidence score
        aps_disp = ''
        for rot_th in thresholds:
            results = compute_match(meta, obj2grasp, obj, pc_id, pred_prefix, rot_th=rot_th, pc_prefix=pc_prefix, topK=topK)
            ap = AP(results, topK=topK)
            aps_disp = aps_disp + 'AP@%d: %.2f | '%(rot_th, ap)
            if not rot_th in APs_at_threshold:
                APs_at_threshold[rot_th] = []
            APs_at_threshold[rot_th].append(ap)
            if not rot_th in APs_for_obj[obj]:
                APs_for_obj[obj][rot_th] = []
            APs_for_obj[obj][rot_th].append(ap)
        pbar.set_description('%s%s | %s'%(aps_disp, obj, pc_id))
        pbar.update(1)
    pbar.close()
    del pbar
    for rot_th in thresholds:
        APs_at_threshold[rot_th] = np.mean(APs_at_threshold[rot_th])
    for obj in list(APs_for_obj):
        for rot_th in thresholds:
            APs_for_obj[obj][rot_th] = np.mean(APs_for_obj[obj][rot_th])
    for obj in sorted(list(APs_for_obj)):
        msg = ''
        for rot_th in thresholds:
            msg = msg + 'mAP@%d: %.4f | '%(rot_th, APs_for_obj[obj][rot_th])
        msg = msg + obj # copy
        print(msg)
    for rot_th in thresholds:
        print('mAP@%d: %.4f'%(rot_th, APs_at_threshold[rot_th]))
    with open(output, 'w') as fp:
        fp.write(json.dumps({'overall': APs_at_threshold, 'object': APs_for_obj}))
