import open3d as o3d
import itertools
import numpy as np
import numba as nb
from pykdtree.kdtree import KDTree
from scipy.spatial import KDTree as sKDTree

def estimate_normals(pc, camera_location, radius=0.003, knn=20):
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc)
    pc_o3d.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=knn))
    pc_o3d.orient_normals_towards_camera_location(camera_location)
    n = np.asarray(pc_o3d.normals, dtype=np.float32)
    return n

@nb.jit(nopython=True, nogil=True)
def force_closure(p1, p2, n1, n2, angle, use_abs_value=True):
    n1, n2 = -n1, -n2 # inward facing normals

    for normal, contact, other_contact in [(n1, p1, p2), (n2, p2, p1)]:
        diff = other_contact - contact
        normal_proj = np.dot(normal, diff) / np.linalg.norm(normal)
        if use_abs_value:
            normal_proj = np.fabs(np.dot(normal, diff)) / np.linalg.norm(normal)

        if normal_proj < 0:
            return 0 # wrong side
        alpha = np.arccos(normal_proj / np.linalg.norm(diff))
        if alpha > angle:
            return 0 # outside of friction cone
    return 1

def reevaluate_antipodal_grasp(crop, pose, max_degree=30.0, hand_height=0.05, gripper_width=0.05, thickness_side=0.01, contact_r=0.003, search_r=0.005, step_size=0.001, viable_th=5, verbose=True):
    # crop: (n, 3)
    # pose: (3, 4)
    max_rad = max_degree / 180.0 * np.pi

    crop = crop.reshape(-1, 3)

    '''
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(crop)
    o3d.io.write_point_cloud("original.ply", pc_o3d)
    '''

    eye = np.eye(4)
    eye[:3,:4] = pose
    pose = eye
    del eye
    crop_c = np.pad(crop.T, ((0,1),(0,0)), mode='constant', constant_values=1.0) # x, y, z, w=1, (4,n)
    inv = np.eye(4)
    inv[:3,:3] = pose[:3,:3].T
    inv[:3,3:4] = -inv[:3,:3]@pose[:3,3:4] # (3, 3) x (3, 1) -> (3, 1)
    crop_c = np.dot(inv, crop_c) # (4,4) x (4,n) -> (4,n)
    crop_c = crop_c[:3].T # (n, 3) -- transformed

    '''
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(crop_c)
    o3d.io.write_point_cloud("tf.ply", pc_o3d)
    '''

    # Crop ROI
    crop_c = crop_c[crop_c[:,0] > 0]
    crop_c = crop_c[crop_c[:,0] < hand_height]
    crop_c = crop_c[crop_c[:,1] > -gripper_width/2.0]
    crop_c = crop_c[crop_c[:,1] <  gripper_width/2.0]
    crop_c = crop_c[crop_c[:,2] > -thickness_side/2.0]
    crop_c = crop_c[crop_c[:,2] <  thickness_side/2.0]

    '''
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(crop_c)
    o3d.io.write_point_cloud("roi.ply", pc_o3d)
    '''

    # sweep along gripper y-axis to simulate closing the gripper
    r_contact = gripper_width/2.0
    while r_contact > 0:
        if (crop_c[:,1] > r_contact).any():
            break
        r_contact -= step_size
    if r_contact < 0:
        return False, 0, 0, 0, 0
    l_contact = -gripper_width/2.0
    while l_contact < 0:
        if (crop_c[:,1] < l_contact).any():
            break
        l_contact += step_size
    if l_contact > 0:
        return False, 0, 0, 0, 0

    # crop contact points
    crop_c = crop_c[(crop_c[:,1] < (l_contact+contact_r)) | (crop_c[:,1] > (r_contact-contact_r))]

    '''
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(crop_c)
    o3d.io.write_point_cloud("contact-pre.ply", pc_o3d)
    '''

    p_l = crop_c[crop_c[:,1]<0].astype(np.float32) # left point cloud
    p_r = crop_c[crop_c[:,1]>0].astype(np.float32) # right point cloud
    if len(p_l) == 0 or len(p_r) == 0:
        return False, 0, 0, 0, 0
    n_l = estimate_normals(p_l, [0, -gripper_width*100, 0]).astype(np.float32) # points outward
    n_r = estimate_normals(p_r, [0,  gripper_width*100, 0]).astype(np.float32) # points outward

    l_frame = p_l - np.array([0, l_contact, 0], dtype=np.float32) # (num_l, 3)
    r_frame = p_r - np.array([0, r_contact, 0], dtype=np.float32) # (num_r, 3)

    l_tree = sKDTree(l_frame)
    index_l = l_tree.query_ball_point(r_frame, search_r, p=2.0)

    match_l = set()
    match_r = set()
    l_contacts = set()
    r_contacts = set()
    n_antipodal_pairs = 0
    for r, ll in enumerate(index_l):
        for l in ll:
            l_contacts.add(l)
            r_contacts.add(r)
            if force_closure(p_l[l], p_r[r], n_l[l], n_r[r], max_rad): # test if force closure
                match_l.add(l)
                match_r.add(r)
                n_antipodal_pairs += 1
    l_viable = len(match_l)
    r_viable = len(match_r)
    viable = min(l_viable, r_viable)
    if verbose:
        print("l_viable: %d, r_viable: %d"%(l_viable, r_viable))
        print("Found %d contant points on the left"%len(l_contacts))
        print("Found %d contant points on the right"%len(r_contacts))
        print("Found %d antipodal pairs on the right"%n_antipodal_pairs)

    '''
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(np.append(p_l[l_contacts], p_r[r_contacts], axis=0))
    o3d.io.write_point_cloud("contact-post.ply", pc_o3d)
    '''

    '''
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(np.append(p_l[match_l], p_r[match_r], axis=0))
    o3d.io.write_point_cloud("antipodal.ply", pc_o3d)
    '''

    return viable>viable_th, l_viable, r_viable, len(l_contacts), len(r_contacts), n_antipodal_pairs

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

def metrics(ps, ts, mAP_topK=10, **kwargs):
    tp, fp, fn, sp = 0, 0, 0, 0
    n_p, n_t = 0, 0
    precisions = []
    ts_matched = set() # ground truth pose set that matched
    ts_kd_tree = KDTree(np.asarray(ts)[:,:,3])
    for n, (score, pose) in enumerate(ps):
        trans_matched = ts_kd_tree.query(pose[:,3].reshape(1, 3), k=len(ts), distance_upper_bound=kwargs['trans_th'])[1][0]
        matched = 0
        for i in trans_matched[trans_matched<len(ts)]:
            if rot_match(pose, ts[i], th=kwargs['rot_th']):
                if i in ts_matched:
                    matched = 1
                else:
                    matched = 2
                    ts_matched.add(i)
                    break
        if matched == 2:
            tp += 1
            if n < mAP_topK:
                precisions.append(tp / (n + 1.))
        elif matched == 1:
            sp += 1
        else:
            fp += 1
    fn += len(ts) - len(ts_matched)
    n_p += len(ps)
    n_t += len(ts)
    ap = np.mean(precisions) if len(precisions)>0 else 0
    return (tp, fp, fn, sp, n_p, n_t, ap)

def batch_metrics(pred_poses, gt_poses, **kwargs):
    tp, fp, fn, sp = 0, 0, 0, 0
    n_p, n_t = 0, 0
    ap = 0
    for ps, ts in zip(pred_poses, gt_poses): # for every object
        (_tp, _fp, _fn, _sp, _n_p, _n_t, _ap) = metrics(ps, ts, **kwargs)
        tp += _tp
        fp += _fp
        fn += _fn
        sp += _sp
        n_p += _n_p
        n_t += _n_t
        ap  += _ap
    return tp, fp, fn, sp, n_p, n_t, ap / float(len(gt_poses))
