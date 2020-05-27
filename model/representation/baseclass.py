from abc import ABC, abstractmethod


class AbstractRepresentation(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def grasp_representation(self, pose, pts, pts_index):
        raise NotImplementedError('You call the abstract method?')

    @abstractmethod
    def recover_grasp(self, pts, xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual):
        raise NotImplementedError('You call the abstract method?')

    @abstractmethod
    def update_feature_volume(self, feature, index, xyz, roll, pitch_index, pitch_residual, yaw_index, yaw_residual):
        raise NotImplementedError('You call the abstract method?')

    @abstractmethod
    def compute_loss(self):
        raise NotImplementedError('You call the abstract method?')
