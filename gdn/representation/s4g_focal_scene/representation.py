from ..baseclass import AbstractRepresentation
from ...utils import rotation_euler, hand_match, generate_gripper_edge, crop_index
import numpy as np
import numba as nb
from scipy.spatial.transform import Rotation


class S4GRepresentation(AbstractRepresentation):
    def __init__(self, config, **kwargs):
        assert isinstance(config, dict)
        super(S4GRepresentation, self).__init__(**kwargs)
        self.config = config
        self.roll_180 = np.array([[1,  0,  0],
                                  [0, -1,  0],
                                  [0,  0, -1]], dtype=np.float32)
    def grasp_representation(self, pose, pts, pts_index):
        # Fix Pose z-axis:
        # z-axis should be always upward wrt to world coordinate to resolve ambigouity of "roll" rotation
        if pose[2,2]<0: # upside-down
            pose[:3,:3] = np.matmul(self.roll_180, pose[:3,:3]) # rotate x-axis(roll) to make z-axis upward

        r_inv = pose[:3,:3].T # (3, 3)
        t = pose[:3,-1:] # (3,1)

        d9_rot = pose[:3,:3].reshape(-1) # (3,3) -> (9,)
        xyz = np.matmul(r_inv, pts[pts_index].T-t).T # translation wrt gripper reference frame

        return (xyz, d9_rot)

    def recover_grasp(self):
        raise NotImplementedError("We don't need this function here.")

    def update_feature_volume(self, feature, index, xyz, d9_rot):
        '''
        feature: (M, 13)
        index: (K,), K<M
        xyz: (K, 3)
        d9_rot: (9,) -- flatten rotation matrix
        '''
        feature[index, 0]     = 1.0
        feature[index, 1:10]  = d9_rot
        feature[index, 10:13] = xyz

    def retrive_from_feature_volume_batch(self, pts, features, **kwargs):
        new_poses = []
        for b in range(len(pts)):

            new_poses.append(retrive_from_feature_volume(pts[b],
                                                         features[b],
                                                         *features[b].shape[:-1],
                                                         self.config['hand_height'],
                                                         self.config['gripper_width'],
                                                         self.config['thickness_side'],
                                                         self.config['rot_th'],
                                                         self.config['trans_th'],
                                                         **kwargs))
        return new_poses

    def filter_out_invalid_grasp_batch(self, pts, poses):
        new_poses = []
        for b in range(len(pts)):
            new_poses.append(self.filter_out_invalid_grasp(pts[b], poses[b]))
        return new_poses

    def filter_out_invalid_grasp(self, pts, poses, n_collision=2):
        for i in reversed(range(len(poses))):
            socre, pose = poses[i]
            gripper_outer1, gripper_outer2 = generate_gripper_edge(self.config['gripper_width']+self.config['thickness']*2, self.config['hand_height'], pose, self.config['thickness_side'], backward=0.20)[1:]
            gripper_inner1, gripper_inner2 = generate_gripper_edge(self.config['gripper_width'], self.config['hand_height'], pose, self.config['thickness_side'])[1:]
            outer_pts = crop_index(pts, gripper_outer1, gripper_outer2)
            if len(outer_pts) == 0:
                del poses[i]
                continue
            inner_pts = crop_index(pts, gripper_inner1, gripper_inner2, search_idx=outer_pts)
            if len(outer_pts) - len(np.intersect1d(inner_pts, outer_pts)) > n_collision:
                del poses[i]
                continue
        return poses


@nb.njit
def retrive_from_feature_volume(pts, feature, M, hand_height, gripper_width, thickness_side, rot_th, trans_th, n_output=10, threshold=0.0, nms=False):
    roll_180 = np.array([[1,  0,  0],
                         [0, -1,  0],
                         [0,  0, -1]], dtype=np.float32)
    if M < n_output:
        n_output = M
    index = np.argsort(feature[:,0])[-n_output:][::-1]
    poses = []
    for pt_idx in index:
        score = feature[pt_idx, 0]
        if score <= threshold:
            break
        rot = feature[pt_idx, 1:10].reshape(3,3) # (9,) -> (3,3)
        if rot[2,2]<0: # upside-down
            rot = np.dot(roll_180, rot) # flip
        xyz = feature[pt_idx:pt_idx+1, 10:13] # (1, 3)
        xyz = xyz.T # (3, 1)
        pt  = pts[pt_idx:pt_idx+1].T # (3, 1)
        tran = pt - np.dot(rot, xyz) # (3, 1) - (3, 3) x (3, 1) = (3, 1)
        pose = np.append(rot, tran, axis=-1).astype(np.float32) # (3, 4)

        # NMS
        if nms:
            matched = False
            for n, (prev_score, prev_pred) in enumerate(poses):
                if hand_match(prev_pred, pose, rot_th=rot_th, trans_th=trans_th):
                    matched = True
                    break
            if matched:
                continue
        poses.append((score, pose))
    return poses