from ..baseclass import AbstractRepresentation
from ...utils import rotation_euler, hand_match, generate_gripper_edge, crop_index
import numpy as np
import numba as nb
from scipy.spatial.transform import Rotation


class EulerNoBinRepresentation(AbstractRepresentation):
    def __init__(self, config, **kwargs):
        assert isinstance(config, dict)
        super(EulerNoBinRepresentation, self).__init__(**kwargs)
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

        '''
        roll: [-pi, +pi]
        pitch: [-pi/2, +pi/2]
        yaw: [-pi, +pi]
        hand_height: (x range)
        gripper_width:  (y range)
        thickness_side: (z range)
        '''

        roll, pitch, yaw = Rotation.from_matrix(pose[:3,:3]).as_euler('xyz', degrees=False).astype(np.float32)
        xyz = np.matmul(r_inv, pts[pts_index].T-t).T
        xyz[...,0] = xyz[...,0] / self.config['hand_height'] # [0, 1]
        xyz[...,1] = (xyz[...,1] + self.config['gripper_width'] / 2.0) / self.config['gripper_width'] # [0, 1]
        xyz[...,2] = (xyz[...,2] + self.config['thickness_side'] / 2.0) / self.config['thickness_side'] # [0, 1]
        xyz = np.clip(xyz, 1e-8, 1-1e-8)

        roll = np.array([np.cos(2*roll), np.sin(2*roll)]) # Gripper is 2-fold symmetric

        return (xyz, roll, pitch, yaw)

    def recover_grasp(self):
        raise NotImplementedError("We don't need this function here.")

    def update_feature_volume(self, feature, index, xyz, roll, pitch, yaw):
        '''
        feature: (M, 8)
        index: (K,), K<M
        xyz: (K, 3)
        roll: (2,)
        pitch: float
        yaw: float
        '''
        # feature in each cell: logit, x, y, z, roll, pitch, yaw
        feature[index, 0] += 1.0   # 0
        feature[index, 1:4] = xyz  # 1,2,3
        feature[index, 4:6] = roll # 4,5
        feature[index, 6] = pitch
        feature[index, 7] = yaw

    def retrive_from_feature_volume_batch(self, pts, indices, features, **kwargs):
        new_poses = []
        for b in range(len(pts)):

            new_poses.append(retrive_from_feature_volume(pts[b][indices[b]],
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
        xyz = feature[pt_idx:pt_idx+1, 1:4] # (1, 3)
        xyz[:,0] =  xyz[:,0] * hand_height
        xyz[:,1] = (xyz[:,1] * gripper_width)  - gripper_width  / 2.0
        xyz[:,2] = (xyz[:,2] * thickness_side) - thickness_side / 2.0
        xyz = xyz.T # (3, 1)
        pt  = pts[pt_idx:pt_idx+1].T # (3, 1)

        roll  = feature[pt_idx,4:6]
        if roll[1] == 0:
            roll[1] = 1e-8

        roll_f = float(np.arctan((1-roll[0])/roll[1]))
        pitch = float(feature[pt_idx,6]) # float
        yaw   = float(feature[pt_idx,7]) # float
        rot = rotation_euler(roll_f, pitch, yaw)
        if rot[2,2]<0: # upside-down
            rot = np.dot(roll_180, rot) # flip
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
