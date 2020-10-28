import torch.nn as nn
import torch.optim as optim
from ..detector.utils import freeze_model, unfreeze_model
from itertools import chain


def import_model_by_setting(config, mode='train'):
    # Load representation
    # Load model output (activation) layer: ActivationLayer
    if config['representation'] == 'euler':
        from ..representation.euler import EulerRepresentation, MultiTaskLossWrapper
        from ..representation.euler.activation import EulerActivation as ActivationLayer
        representation = EulerRepresentation(config)
        loss = MultiTaskLossWrapper(config).cuda()
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            freeze_model(loss)
        if mode == 'train':
            from ..representation.euler import GraspDataset, collate_fn_setup
            dataset = GraspDataset(config)
            my_collate_fn = collate_fn_setup(config, representation)
        else:
            from ..representation.euler import GraspDatasetVal, val_collate_fn_setup
            dataset = GraspDatasetVal(config)
            my_collate_fn = val_collate_fn_setup(config)
    elif config['representation'] == 'euler_scene':
        from ..representation.euler_scene import EulerRepresentation, MultiTaskLossWrapper
        from ..representation.euler_scene.activation import EulerActivation as ActivationLayer
        representation = EulerRepresentation(config)
        loss = MultiTaskLossWrapper(config).cuda()
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            freeze_model(loss)
        if mode == 'train':
            from ..representation.euler_scene import GraspDataset, collate_fn_setup
            dataset = GraspDataset(config)
            my_collate_fn = collate_fn_setup(config, representation)
        else:
            from ..representation.euler_scene import GraspDatasetVal, collate_fn_setup_val
            dataset = GraspDatasetVal(config)
            my_collate_fn = collate_fn_setup_val(config)
    elif config['representation'] == 'euler_scene_att':
        from ..representation.euler_scene_att import EulerRepresentation, MultiTaskLossWrapper
        from ..representation.euler_scene_att.activation import EulerActivation as ActivationLayer
        representation = EulerRepresentation(config)
        loss = MultiTaskLossWrapper(config).cuda()
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            freeze_model(loss)
        if mode == 'train':
            from ..representation.euler_scene_att import GraspDataset, collate_fn_setup
            dataset = GraspDataset(config)
            my_collate_fn = collate_fn_setup(config, representation)
        else:
            from ..representation.euler_scene_att import GraspDatasetVal, collate_fn_setup_val
            dataset = GraspDatasetVal(config)
            my_collate_fn = collate_fn_setup_val(config)
    elif config['representation'] == 'euler_scene_rp':
        from ..representation.euler_scene_rp import EulerRepresentation, MultiTaskLossWrapper
        from ..representation.euler_scene_rp.activation import EulerActivation as ActivationLayer
        representation = EulerRepresentation(config)
        loss = MultiTaskLossWrapper(config).cuda()
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            freeze_model(loss)
        if mode == 'train':
            from ..representation.euler_scene_rp import GraspDataset, collate_fn_setup
            dataset = GraspDataset(config)
            my_collate_fn = collate_fn_setup(config, representation)
        else:
            from ..representation.euler_scene_rp import GraspDatasetVal, collate_fn_setup_val
            dataset = GraspDatasetVal(config)
            my_collate_fn = collate_fn_setup_val(config)
    elif config['representation'] == 'euler_noisy':
        from ..representation.euler_noisy import EulerRepresentation
        from ..representation.euler_noisy import EulerActivation as ActivationLayer
        representation = EulerRepresentation(config)
        loss = None
        if mode == 'train':
            raise NotImplementedError("No traning code for euler_noisy! Please use 'euler' representation instead!")
        else:
            from ..representation.euler_noisy import GraspDatasetVal, val_collate_fn_setup
            dataset = GraspDatasetVal(config)
            my_collate_fn = val_collate_fn_setup(config)
    elif config['representation'] == 'euler_asymmetric_roll':
        from ..representation.euler_asymmetric_roll import EulerRepresentation, MultiTaskLossWrapper
        from ..representation.euler_asymmetric_roll.activation import EulerActivation as ActivationLayer
        representation = EulerRepresentation(config)
        loss = MultiTaskLossWrapper(config).cuda()
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            freeze_model(loss)
        if mode == 'train':
            from ..representation.euler_asymmetric_roll import GraspDataset, collate_fn_setup
            dataset = GraspDataset(config)
            my_collate_fn = collate_fn_setup(config, representation)
        else:
            from ..representation.euler_asymmetric_roll import GraspDatasetVal, val_collate_fn_setup
            dataset = GraspDatasetVal(config)
            my_collate_fn = val_collate_fn_setup(config)
    elif config['representation'] == 'euler_regression':
        from ..representation.euler_regression import EulerRegressionRepresentation, MultiTaskLossWrapper
        from ..representation.euler_regression.activation import EulerRegressionActivation as ActivationLayer
        representation = EulerRegressionRepresentation(config)
        loss = MultiTaskLossWrapper(config).cuda()
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            freeze_model(loss)
        if mode == 'train':
            from ..representation.euler_regression import GraspDataset, collate_fn_setup
            dataset = GraspDataset(config)
            my_collate_fn = collate_fn_setup(config, representation)
        else:
            from ..representation.euler_regression import GraspDatasetVal, val_collate_fn_setup
            dataset = GraspDatasetVal(config)
            my_collate_fn = val_collate_fn_setup(config)
    elif config['representation'] == 'euler_no_bin':
        from ..representation.euler_no_bin import EulerNoBinRepresentation, MultiTaskLossWrapper
        from ..representation.euler_no_bin.activation import EulerNoBinActivation as ActivationLayer
        representation = EulerNoBinRepresentation(config)
        loss = MultiTaskLossWrapper(config).cuda()
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            freeze_model(loss)
        if mode == 'train':
            from ..representation.euler_no_bin import GraspDataset, collate_fn_setup
            dataset = GraspDataset(config)
            my_collate_fn = collate_fn_setup(config, representation)
        else:
            from ..representation.euler_no_bin import GraspDatasetVal, val_collate_fn_setup
            dataset = GraspDatasetVal(config)
            my_collate_fn = val_collate_fn_setup(config)
    elif config['representation'] == 's4g':
        from ..representation.s4g import S4GRepresentation, MultiTaskLossWrapper
        from ..representation.s4g.activation import S4GActivation as ActivationLayer
        representation = S4GRepresentation(config)
        loss = MultiTaskLossWrapper(config).cuda()
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            freeze_model(loss)
        if mode == 'train':
            from ..representation.s4g import GraspDataset, collate_fn_setup
            dataset = GraspDataset(config)
            my_collate_fn = collate_fn_setup(config, representation)
        else:
            from ..representation.s4g import GraspDatasetVal, val_collate_fn_setup
            dataset = GraspDatasetVal(config)
            my_collate_fn = val_collate_fn_setup(config)
    elif config['representation'] == 's4g_focal':
        from ..representation.s4g_focal import S4GRepresentation, MultiTaskLossWrapper
        from ..representation.s4g_focal.activation import S4GActivation as ActivationLayer
        representation = S4GRepresentation(config)
        loss = MultiTaskLossWrapper(config).cuda()
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            freeze_model(loss)
        if mode == 'train':
            from ..representation.s4g_focal import GraspDataset, collate_fn_setup
            dataset = GraspDataset(config)
            my_collate_fn = collate_fn_setup(config, representation)
        else:
            from ..representation.s4g_focal import GraspDatasetVal, val_collate_fn_setup
            dataset = GraspDatasetVal(config)
            my_collate_fn = val_collate_fn_setup(config)
    else:
        raise NotImplementedError("Your setting is invalid! Please check the configuration file and try again.")

    # Instancelize ActivationLayer
    model_output_layer = ActivationLayer()
    model_output_layer = model_output_layer.cuda() # Since there is no weights in this layer. Maybe we can remove this line?

    # Load model backbone
    if config['backbone'] == 'pointnet2':
        from ..detector.pointnet2.backbone import Pointnet2MSG
        base_model = Pointnet2MSG(config, activation_layer=model_output_layer).cuda()
    elif config['backbone'] == 'pointnet2_s4g':
        from ..detector.pointnet2_s4g.backbone import Pointnet2MSG
        base_model = Pointnet2MSG(config, activation_layer=model_output_layer).cuda()
    elif config['backbone'] == 'pointnet2_att':
        from ..detector.pointnet2_att.backbone import Pointnet2MSG
        base_model = Pointnet2MSG(config, activation_layer=model_output_layer).cuda()
    elif config['backbone'] == 'edgeconv':
        from ..detector.edgeconv.backbone import EdgeDet
        base_model = EdgeDet(config, activation_layer=model_output_layer).cuda()
    else:
        raise NotImplementedError("No such backbone. Please check the configuration file and try again.")
    parallel_model = nn.DataParallel(base_model)

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(chain(parallel_model.parameters(), loss.parameters()), lr=config['learning_rate'])
    else:
        raise NotImplementedError("Not support %s now."%config['optimizer'])

    return representation, dataset, my_collate_fn, base_model, parallel_model, optimizer, loss

