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
from argparse import ArgumentParser

warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)
warnings.filterwarnings('ignore', category=nb.NumbaPerformanceWarning)

config = dict(
    hand_height = 0.11, # 11cm (x range)
    gripper_width = 0.08, # 8cm (y range)
    thickness_side = 0.01, # (z range)
    thickness = 0.01, # gripper_width + thickness*2 = hand_outer_diameter (0.08+0.01*2 = 0.1)
)

def generate_gripper_edge(width, hand_height, gripper_pose_wrt_mass_center, thickness, backward=0.0):
    pad = np.array([[0.,0.,0.,1.]]*4, dtype=np.float32)
    gripper_r = np.array([0.,1.,0.], dtype=np.float32)*width/2.
    gripper_l = -gripper_r
    gripper_l_t = gripper_l + np.array([hand_height,0.,0.], dtype=np.float32)
    gripper_r_t = gripper_r + np.array([hand_height,0.,0.], dtype=np.float32)
    gripper_r[0] -= backward
    gripper_l[0] -= backward
    thickness_d = np.array([0., 0., thickness/2.], dtype=np.float32).reshape(1, 3)

    gripper = np.stack( (gripper_l, gripper_r, gripper_l_t, gripper_r_t) ) # (4, 3)
    gripper_outer1 = gripper - thickness_d
    gripper_outer2 = gripper + thickness_d

    # gripper = np.pad(gripper, ((0,0), (0,1)), mode='constant', constant_values=1)
    pad[:,:3] = gripper
    gripper = np.dot(gripper_pose_wrt_mass_center, pad.T).T

    pad[:,:3] = gripper_outer1
    gripper_outer1 = np.dot(gripper_pose_wrt_mass_center, pad.T).T

    pad[:,:3] = gripper_outer2
    gripper_outer2 = np.dot(gripper_pose_wrt_mass_center, pad.T).T

    return gripper, gripper_outer1, gripper_outer2


@nb.jit(nopython=True, nogil=True)
def hand_match(pred, target, rot_th=5, trans_th=0.02):
    # pred: (3, 4)
    # target: (3, 4)
    rot_th_rad = np.radians(rot_th)
    trans_matched = np.linalg.norm(pred[:,3]-target[:,3], ord=2) < trans_th
    rot_diff = np.arcsin(np.linalg.norm( np.eye(3) - np.dot(pred[:3,:3], target[:3,:3].T), ord=None  ) / (2*np.sqrt(2)))
    rot_matched = rot_diff < rot_th_rad
    return rot_matched and trans_matched

@nb.jit(nopython=True, nogil=True)
def rot_match(pred, target, th=5):
    rot_th_rad = np.radians(th)
    rot_diff = np.arcsin(np.linalg.norm( np.eye(3) - np.dot(pred[:3,:3], target[:3,:3].T), ord=None  ) / (2*np.sqrt(2)))
    return rot_diff < rot_th_rad

def compute_match(ps, ts, predict_prefix='', pc_prefix=None, rot_th=5.0, trans_th=0.02):
    ts_kd_tree = {k: KDTree(np.asarray(v)[:,:,3]) for (k, v) in ts.items() } # KD-tree for each object
    ts_matched = set() # ground truth pose set that matched
    results = []
    for score, obj_id, pc_id, row_n in ps:
        pose = np.load(predict_prefix+'/'+obj_id+'/'+pc_id+'.npy')[row_n] # load corresponding prediction ordered by confidence

        trans_matched = ts_kd_tree[obj_id].query(pose[:,3].reshape(1, 3), k=len(ts[obj_id]), distance_upper_bound=trans_th)[1][0]
        matched = False
        for i in trans_matched[trans_matched<len(ts[obj_id])]:
            if rot_match(pose, ts[obj_id][i], th=rot_th) and not (obj_id, pc_id, i) in ts_matched:
                matched = True
                ts_matched.add((obj_id, pc_id, i))
                break
        results.append(matched)
        if not pc_prefix is None:
            pc_npy = np.load(pc_prefix+'/'+obj_id+'/clouds/'+pc_id+'.npy') # load corresponding prediction ordered by confidence
            if len(pc_npy)>1000:
                pc_npy = pc_npy[np.random.choice(len(pc_npy), 1000, replace=False)]
            mlab.clf()
            mlab.points3d(pc_npy[:,0], pc_npy[:,1], pc_npy[:,2], scale_factor=0.004, mode='sphere', color=(1.0,1.0,0.0), opacity=1.0)
            gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
            gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge

            mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=config['thickness']/4., color=(0,0,1) if matched else (1,0,0), opacity=0.5)
            mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=config['thickness']/4., color=(0,0,1) if matched else (1,0,0), opacity=0.5)
            mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=config['thickness']/4., color=(0,0,1) if matched else (1,0,0), opacity=0.5)
            if matched:
                gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], ts[obj_id][i], config['thickness_side'])
                gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge

                mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=config['thickness']/4., color=(0,1,0), opacity=0.5)
                mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=config['thickness']/4., color=(0,1,0), opacity=0.5)
                mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=config['thickness']/4., color=(0,1,0), opacity=0.5)
            mlab.show()
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
    parser.add_argument("--pc_path", type=str, default="", help="Point cloud path for visualization")
    parser.add_argument("--top_K", type=int, default=10, help="Top-K for AP computation [10]")
    args = parser.parse_args()

    gt_prefix = args.gt
    pred_prefix = args.pred
    suffix = args.gt_suffix
    pc_prefix = None
    if os.path.exists(args.pc_path):
        from mayavi import mlab
        pc_prefix = args.pc_path
    topK = args.top_K
    gt_id = os.listdir(gt_prefix)
    pred_files = glob.glob(pred_prefix + '/*/*.meta')
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
        if not obj in APs_for_obj:
            APs_for_obj[obj] = {}
        with open(path, 'rb') as fp:
            meta = list(filter(lambda x: x[0]!=-np.inf, pickle.load(fp)))
            meta.sort(reverse=True) # sort by confidence score
        aps_disp = ''
        for rot_th in thresholds:
            results = compute_match(meta, obj2grasp, pred_prefix, rot_th=rot_th, pc_prefix=pc_prefix)
            ap = AP(results, topK=topK)
            aps_disp = aps_disp + 'AP@%d: %.2f | '%(rot_th, ap)
            if not rot_th in APs_at_threshold:
                APs_at_threshold[rot_th] = []
            APs_at_threshold[rot_th].append(ap)
            if not rot_th in APs_for_obj[obj]:
                APs_for_obj[obj][rot_th] = []
            APs_for_obj[obj][rot_th].append(ap)
        pbar.set_description('%s%s'%(aps_disp, obj))
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
    with open('eval.json', 'w') as fp:
        fp.write(json.dumps({'overall': APs_at_threshold, 'object': APs_for_obj}))
