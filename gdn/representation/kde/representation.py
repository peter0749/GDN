from ..baseclass import AbstractRepresentation
from ...utils import rotation_euler, hand_match, generate_gripper_edge, crop_index
import numpy as np
import numba as nb
from scipy.spatial.transform import Rotation


class KDERepresentation(AbstractRepresentation):
    def __init__(self, config, **kwargs):
        assert isinstance(config, dict)
        super(KDERepresentation, self).__init__(**kwargs)
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

        d6_rot = pose[:3,:2].reshape(-1)
        xyz = np.matmul(r_inv, pts[pts_index].T-t).T # translation wrt gripper reference frame

        return (xyz, d6_rot)

    def recover_grasp(self):
        raise NotImplementedError("We don't need this function here.")

    def update_feature_volume(self):
        raise NotImplementedError("We don't need this function here.")

    def retrive_from_feature_volume_batch(self, poses, **kwargs):
        raise NotImplementedError("Not implement yet!")

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
