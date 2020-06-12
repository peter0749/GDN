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

config = dict(
    hand_height = 0.11, # 11cm (x range)
    gripper_width = 0.08, # 8cm (y range)
    thickness_side = 0.01, # (z range)
    thickness = 0.01, # gripper_width + thickness*2 = hand_outer_diameter (0.08+0.01*2 = 0.1)
)

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

def compute_match(ps, ts, obj_id, pc_id, predict_prefix='', pc_prefix=None, rot_th=5.0, trans_th=0.02, topK=10, vis_conf=False, vis_gamma=0.6, vis_proposal=False, use_mayavi=False):
    if not pc_prefix is None:
        plots = [] # for plotly
        if _VISUALIZE_USE_PCL:
            pc = pcl.PointCloud_PointXYZRGB()
            pc.from_file((pc_prefix+'/'+obj_id+'/clouds/'+pc_id+'.pcd').encode())
            pc_npy = np.array(pc)[:,:3]
            if not pc_prefix is None:
                if use_mayavi:
                    fig = mlab.figure(bgcolor=(1,1,1))
                    pc_npy = np.load(pc_prefix+'/'+obj_id+'/clouds/'+pc_id+'.npy') # load corresponding prediction ordered by confidence
                    mlab.points3d(pc_npy[:,0], pc_npy[:,1], pc_npy[:,2], scale_factor=0.0015, scale_mode='none', mode='sphere', color=(0.7,0.7,0.7), opacity=1.0, figure=fig)
                else:
                    rgb = np.array([ decode_pcl_rgb(pc[i][3]) for i in range(pc.size) ], dtype=np.float32)
                    points_scatter_plot = go.Scatter3d(
                        x = pc_npy[:,0],
                        y = pc_npy[:,1],
                        z = pc_npy[:,2],
                        mode = 'markers',
                        marker=dict(
                            size=2.0,
                            color=rgb,
                            opacity=1.0
                        )
                    )
                    plots.append(points_scatter_plot)
            del pc
        else:
            pc_npy = np.load(pc_prefix+'/'+obj_id+'/clouds/'+pc_id+'.npy') # load corresponding prediction ordered by confidence
            if not pc_prefix is None:
                if use_mayavi:
                    fig = mlab.figure(bgcolor=(1,1,1))
                    pc_npy = np.load(pc_prefix+'/'+obj_id+'/clouds/'+pc_id+'.npy') # load corresponding prediction ordered by confidence
                    mlab.points3d(pc_npy[:,0], pc_npy[:,1], pc_npy[:,2], scale_factor=0.0015, scale_mode='none', mode='sphere', color=(0.7,0.7,0.7), opacity=1.0, figure=fig)
                else:
                    points_scatter_plot = go.Scatter3d(
                        x = pc_npy[:,0],
                        y = pc_npy[:,1],
                        z = pc_npy[:,2],
                        mode = 'markers',
                        marker=dict(
                            size=2.0,
                            color='gray',
                            opacity=1.0
                        )
                    )
                    plots.append(points_scatter_plot)
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
        if (not pc_prefix is None) and n_cnt<topK:
            gripper_inner_edge, gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])
            gripper_l, gripper_r, gripper_l_t, gripper_r_t = gripper_inner_edge
            center_bottom = (gripper_l+gripper_r) / 2.0
            approach = gripper_l_t-gripper_l
            approach = approach / np.linalg.norm(approach) # norm must > 0
            wrist_center = center_bottom - approach * 0.05

            if vis_conf:
                control_p = np.clip((n_cnt/topK)**vis_gamma, 0, 1)
                green_hsv = matplotlib.colors.rgb_to_hsv(np.array([0, 100, 0]))
                red_hsv   = matplotlib.colors.rgb_to_hsv(np.array([255, 0, 0]))
                s = np.asarray(np.clip(matplotlib.colors.hsv_to_rgb(control_p * red_hsv + (1.-control_p) * green_hsv), 0, 255), dtype=np.uint8)
            elif vis_proposal:
                s = (100,100,100,178)
            else:
                s = (0,100,0,178) if matched else (255,0,0,178)

            if not pc_prefix is None:
                if use_mayavi:
                    s = tuple(np.asarray(s[:3], dtype=np.float32) / 255.0)
                    mlab.plot3d([gripper_l[0], gripper_r[0]], [gripper_l[1], gripper_r[1]], [gripper_l[2], gripper_r[2]], tube_radius=0.003, color=s, opacity=0.8, figure=fig)
                    mlab.plot3d([gripper_l[0], gripper_l_t[0]], [gripper_l[1], gripper_l_t[1]], [gripper_l[2], gripper_l_t[2]], tube_radius=0.003, color=s, opacity=0.8, figure=fig)
                    mlab.plot3d([gripper_r[0], gripper_r_t[0]], [gripper_r[1], gripper_r_t[1]], [gripper_r[2], gripper_r_t[2]], tube_radius=0.003, color=s, opacity=0.8, figure=fig)
                    mlab.plot3d([center_bottom[0], wrist_center[0]], [center_bottom[1], wrist_center[1]], [center_bottom[2], wrist_center[2]], tube_radius=0.003, color=s, opacity=0.8, figure=fig)
                else:
                    s = 'rgba(%d,%d,%d,%d)'%s
                    plots.append(
                        plot_gripper_tube(gripper_l, gripper_r, color=s)
                    )
                    plots.append(
                        plot_gripper_tube(gripper_l, gripper_l_t, color=s)
                    )
                    plots.append(
                        plot_gripper_tube(gripper_r, gripper_r_t, color=s)
                    )
                    plots.append(
                        plot_gripper_tube(center_bottom, wrist_center, color=s)
                    )
    if not pc_prefix is None:
        if use_mayavi:
            mlab.show()
            mlab.close(all=True)
        else:
            fig = go.Figure({
                "data": plots,
                "layout": go.Layout(title="",
                                    showlegend=False,
                                    plot_bgcolor='rgba(1,1,1,1)')
            })
            fig.update_layout(
                scene = dict(
                    xaxis = {
                        'showgrid': False,
                        'zeroline': False,
                        'visible': False,
                    },
                    yaxis = {
                        'showgrid': False,
                        'zeroline': False,
                        'visible': False,
                    },
                    zaxis = {
                        'showgrid': False,
                        'zeroline': False,
                        'visible': False,
                    },
                    aspectratio = dict(x=1, y=1, z=1)
                )
            )
            fig.show()
            input("Press Enter to continue")
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
    parser.add_argument("--visualize_confidence", action="store_true", help="Visualize confidence of grasps")
    parser.add_argument("--visualize_proposal", action="store_true", help="")
    parser.add_argument("--visualize_gamma", type=float, default=1.0, help="")
    parser.add_argument("--use_mayavi", action="store_true", help="Visualize point clouds by using mayavi")
    args = parser.parse_args()

    gt_prefix = args.gt
    pred_prefix = args.pred
    suffix = args.gt_suffix
    output = args.output
    pc_prefix = None
    if os.path.exists(args.pc_path):
        if args.use_mayavi:
            from mayavi import mlab
        else:
            import plotly
            import plotly.graph_objs as go
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
            results = compute_match(meta, obj2grasp, obj, pc_id, pred_prefix, rot_th=rot_th, pc_prefix=pc_prefix, topK=topK, vis_conf=args.visualize_confidence, vis_gamma=args.visualize_gamma, vis_proposal=args.visualize_proposal, use_mayavi=args.use_mayavi)
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
