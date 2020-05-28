import numpy as np
import numba as nb
from pykdtree.kdtree import KDTree


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
