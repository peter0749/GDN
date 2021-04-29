from ..baseclass import AbstractRepresentation
from ...utils import rotation_euler, hand_match, generate_gripper_edge, crop_index
import numpy as np
import numba as nb
from scipy.spatial.transform import Rotation


class EulerRepresentation(AbstractRepresentation):
    def __init__(self, config, **kwargs):
        assert isinstance(config, dict)
        super(EulerRepresentation, self).__init__(**kwargs)
        self.config = config
        self.roll_180 = np.array([[1,  0,  0],
                                  [0, -1,  0],
                                  [0,  0, -1]], dtype=np.float32)
    def grasp_representation(self, pose, pts, pts_index):
        n_pitch, n_yaw = self.config['n_pitch'], self.config['n_yaw']

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

        roll, pitch, yaw = Rotation.from_matrix(pose[:3,:3]).as_euler('xyz', degrees=False)
        pitch_index = int(np.clip((pitch+np.pi/2.0)/np.pi*n_pitch , 0, n_pitch-1))
        yaw_index = int(np.clip((yaw+np.pi)/(2.0*np.pi)*n_yaw, 0, n_yaw-1))
        pitch_residual = (pitch - (pitch_index*(np.pi/n_pitch)-np.pi/2.0)) / (np.pi/n_pitch)
        yaw_residual = (yaw - (yaw_index*(np.pi*2.0/n_yaw)-np.pi)) / (np.pi*2.0/n_yaw)
        xyz = np.matmul(r_inv, pts[pts_index].T-t).T

        xyz[...,0] = xyz[...,0] / self.config['hand_height'] # [0, 1]
        xyz[...,1] = (xyz[...,1] + self.config['gripper_width'] / 2.0) / self.config['gripper_width'] # [0, 1]
        xyz[...,2] = (xyz[...,2] + self.config['thickness_side'] / 2.0) / self.config['thickness_side'] # [0, 1]
        xyz = np.clip(xyz, 1e-8, 1-1e-8)

        # roll of gripper is 2-fold symmertric
        # r = [cos(2r), sin(2r)]
        roll = np.array([np.cos(2*roll), np.sin(2*roll)])

        return (xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual)


    def recover_grasp(self, pts, xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual):
        n_pitch, n_yaw = self.config['n_pitch'], self.config['n_yaw']
        #xyz = np.pad(xy, ((0,0),(0,1)), mode='constant', constant_values=0.0)[...,np.newaxis] # (n, 3, 1)

        xyz[...,0] = xyz[...,0] * self.config['hand_height']
        xyz[...,1] = (xyz[...,1] * self.config['gripper_width']) - self.config['gripper_width'] / 2.0
        xyz[...,2] = (xyz[...,2] * self.config['thickness_side']) - self.config['thickness_side'] / 2.0
        xyz = xyz[...,np.newaxis]

        # r_p = arctan((1-D[0])/D[1]))
        roll[roll[...,1]==0, 1] = 1e-8
        roll = np.arctan((1-roll[...,0])/roll[...,1])

        pitch = pitch_index*(np.pi/n_pitch)-np.pi/2.0 + pitch_residual*(np.pi/n_pitch)
        yaw = yaw_index*(np.pi*2.0/n_yaw)-np.pi + yaw_residual*(np.pi*2.0/n_yaw)
        rot = Rotation.from_euler('xyz',
                            np.concatenate((roll[...,np.newaxis],
                                            pitch[...,np.newaxis],
                                            yaw[...,np.newaxis]), axis=-1),
                            degrees=False).as_matrix() # (N, 3, 3)
        upside_down = rot[:,2,2]<0 # (N,)
        upside_down_num = np.sum(upside_down)
        if upside_down_num>0: # upside-down
            rot[upside_down] = np.matmul(
                np.repeat(
                    self.roll_180[np.newaxis],
                    upside_down_num, axis=0),  # (N, 3, 3)
                rot[upside_down]) # (N, 3, 3) x (N, 3, 3) -> (N, 3, 3)
        trans = pts[...,np.newaxis] - np.matmul(rot, xyz) # (N, 3, 1)
        poses = np.append(rot, trans, axis=-1) # (N, 3, 4)
        return poses

    def update_feature_volume(self, feature, index, xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual):
        '''
        feature: (M, n_pitch, n_yaw, 8)
        index: (K,), K<M
        xyz: (K, 3)
        roll: float
        pitch_index: integer
        pitch_residual: float
        yaw_index: integer
        yaw_residual: float
        '''
        # feature in each cell: logit, x, y, z, roll, pitch_residual, yaw_residual
        feature[index, pitch_index, yaw_index, 0] += 1.0 # 0
        feature[index, pitch_index, yaw_index, 1:4] = xyz # 1,2,3
        feature[index, pitch_index, yaw_index, 4:6] = roll # 4, 5
        feature[index, pitch_index, yaw_index, 6] = pitch_residual
        feature[index, pitch_index, yaw_index, 7] = yaw_residual

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

    def filter_out_invalid_grasp_batch(self, pts, poses, n_collision=2):
        new_poses = []
        for b in range(len(pts)):
            new_poses.append(self.filter_out_invalid_grasp(pts[b], poses[b], n_collision=n_collision))
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
def retrive_from_feature_volume(pts, feature, M, n_pitch, n_yaw, hand_height, gripper_width, thickness_side, rot_th, trans_th, n_output=10, threshold=0.0, nms=False):
    roll_180 = np.array([[1,  0,  0],
                         [0, -1,  0],
                         [0,  0, -1]], dtype=np.float32)
    if M*n_pitch*n_yaw < n_output:
        n_output = M*n_pitch*n_yaw
    index = np.argsort(feature[...,0].ravel())[-n_output:][::-1]
    poses = []
    for ind in index:
        yaw_idx = ind%n_yaw
        pitch_idx = (ind//n_yaw)%n_pitch
        pt_idx = (ind//(n_yaw*n_pitch))%M
        score = feature[pt_idx, pitch_idx, yaw_idx, 0]
        if score <= threshold:
            break
        xyz = feature[pt_idx, pitch_idx, yaw_idx:yaw_idx+1, 1:4] # (1, 3)
        xyz[:,0] = xyz[:,0] * hand_height
        xyz[:,1] = (xyz[:,1] * gripper_width) - gripper_width / 2.0
        xyz[:,2] = (xyz[:,2] * thickness_side) - thickness_side / 2.0
        xyz = xyz.T # (3, 1)
        pt  = pts[pt_idx:pt_idx+1].T # (3, 1)
        roll = feature[pt_idx, pitch_idx, yaw_idx, 4:6] # (2,)
        if roll[1] == 0:
            roll[1] = 1e-8
        roll_f = float(np.arctan((1-roll[0])/roll[1]))
        pitch_residual = feature[pt_idx, pitch_idx, yaw_idx, 6]
        yaw_residual = feature[pt_idx, pitch_idx, yaw_idx, 7]
        pitch = float(pitch_idx*(np.pi/n_pitch)-np.pi/2.0 + pitch_residual*(np.pi/n_pitch))
        yaw = float(yaw_idx*(np.pi*2.0/n_yaw)-np.pi + yaw_residual*(np.pi*2.0/n_yaw))
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

@nb.njit
def retrive_from_feature_volume_fast(pts, feature, M, n_pitch, n_yaw, hand_height, gripper_width, thickness_side, rot_th, trans_th, n_output=10, threshold=0.0, nms=False):
    roll_180 = np.array([[1,  0,  0],
                         [0, -1,  0],
                         [0,  0, -1]], dtype=np.float32)
    if M*n_pitch*n_yaw < n_output:
        n_output = M*n_pitch*n_yaw
    index = np.argsort(feature[...,0].ravel())[-n_output:][::-1]
    poses = []
    for ind in index:
        yaw_idx = ind%n_yaw
        pitch_idx = (ind//n_yaw)%n_pitch
        pt_idx = (ind//(n_yaw*n_pitch))%M
        score = feature[pt_idx, pitch_idx, yaw_idx, 0]
        if score <= threshold:
            break
        xyz = feature[pt_idx, pitch_idx, yaw_idx:yaw_idx+1, 1:4] # (1, 3)
        xyz[:,0] = xyz[:,0] * hand_height
        xyz[:,1] = (xyz[:,1] * gripper_width) - gripper_width / 2.0
        xyz[:,2] = (xyz[:,2] * thickness_side) - thickness_side / 2.0
        xyz = xyz.T # (3, 1)
        pt  = pts[pt_idx:pt_idx+1].T # (3, 1)
        roll = feature[pt_idx, pitch_idx, yaw_idx, 4:6] # (2,)
        if roll[1] == 0:
            roll[1] = 1e-8
        roll_f = float(np.arctan((1-roll[0])/roll[1]))
        pitch_residual = feature[pt_idx, pitch_idx, yaw_idx, 6]
        yaw_residual = feature[pt_idx, pitch_idx, yaw_idx, 7]
        pitch = float(pitch_idx*(np.pi/n_pitch)-np.pi/2.0 + pitch_residual*(np.pi/n_pitch))
        yaw = float(yaw_idx*(np.pi*2.0/n_yaw)-np.pi + yaw_residual*(np.pi*2.0/n_yaw))
        rot = rotation_euler(roll_f, pitch, yaw)
        if rot[2,2]<0: # upside-down
            rot = np.dot(roll_180, rot) # flip
        tran = pt - np.dot(rot, xyz) # (3, 1) - (3, 3) x (3, 1) = (3, 1)
        pose = np.append(rot, tran, axis=-1).astype(np.float32) # (3, 4)
        poses.append(pose)
    return poses

def filter_out_invalid_grasp_fast(config, pts, poses, n_collision=2):
    to_delete = []
    for i in range(len(poses)):
        pose = poses[i]
        gripper_outer1, gripper_outer2 = generate_gripper_edge(config['gripper_width']+config['thickness']*2, config['hand_height'], pose, config['thickness_side'], backward=0.20)[1:]
        gripper_inner1, gripper_inner2 = generate_gripper_edge(config['gripper_width'], config['hand_height'], pose, config['thickness_side'])[1:]
        outer_pts = crop_index(pts, gripper_outer1, gripper_outer2)
        if len(outer_pts) == 0:
            to_delete.append(i)
            continue
        inner_pts = crop_index(pts, gripper_inner1, gripper_inner2, search_idx=outer_pts)
        if len(outer_pts) - len(np.intersect1d(inner_pts, outer_pts)) > n_collision:
            to_delete.append(i)
            continue
    np.delete(poses, to_delete, axis=0)
    return poses
