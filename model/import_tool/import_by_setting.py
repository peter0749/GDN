import torch.nn as nn
import torch.optim as optim
from ..detector.utils import freeze_model, unfreeze_model


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
    else:
        raise NotImplementedError("Your setting is invalid! Please check the configuration file and try again.")

    # Instancelize ActivationLayer
    model_output_layer = ActivationLayer()
    model_output_layer = model_output_layer.cuda() # Since there is no weights in this layer. Maybe we can remove this line?

    # Load model backbone
    if config['backbone'] == 'pointnet2':
        from ..detector.pointnet2.backbone import Pointnet2MSG
        base_model = Pointnet2MSG(config, activation_layer=model_output_layer).cuda()
    elif config['backbone'] == 'edgeconv':
        from ..detector.edgeconv.backbone import EdgeDet
        base_model = EdgeDet(config, activation_layer=model_output_layer).cuda()
    else:
        raise NotImplementedError("No such backbone. Please check the configuration file and try again.")
    parallel_model = nn.DataParallel(base_model)

    if config['optimizer'] == 'adam':
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            optimizer = optim.Adam(parallel_model.parameters(), lr=config['learning_rate'])
        else:
            optimizer = optim.Adam(chain(parallel_model.parameters(), loss.parameters()), lr=config['learning_rate'])
    else:
        raise NotImplementedError("Not support %s now."%config['optimizer'])

    return representation, dataset, my_collate_fn, base_model, parallel_model, optimizer, loss

