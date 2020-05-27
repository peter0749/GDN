from .utils import *
from .detector.utils import *

def import_model_by_setting(config):
    # Load representation
    if config['representation'] == 'euler':
        from .representation.euler import EulerRepresentation, GraspDataset, collate_fn_setup # Contain loss_func
        from .detector.edgeconv.backbone import *
        from .detector.utils import *
        representation = EulerRepresentation(config)
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            representation.freeze_loss_layer()
        dataset = GraspDataset(config)
        my_collate_fn = collate_fn_setup(config, representation)
    else:
        raise NotImplementedError("Your setting is invalid! Please check the configuration file and try again.")

    # Load model backbone
    if config['backbone'] == 'pointnet2':
        from model.detector.pointnet2.backbone import *
        base_model = Pointnet2MSG(config).cuda() # CUDA is required
    elif config['backbone'] == 'edgeconv':
        from model.detector.edgeconv.backbone import *
        base_model = EdgeDet(config).cuda()
    else:
        raise NotImplementedError("No such backbone. Please check the configuration file and try again.")
    parallel_model = nn.DataParallel(base_model)

    if config['optimizer'] == 'adam':
        if not ('tune_task_weights' in config and config['tune_task_weights']):
            optimizer = optim.Adam(parallel_model.parameters(), lr=config['learning_rate'])
        else:
            optimizer = optim.Adam(chain(parallel_model.parameters(), representation.loss_object.parameters()), lr=config['learning_rate'])
    else:
        raise NotImplementedError("Not support %s now."%config['optimizer'])

    return representation, dataset, my_collate_fn, base_model, parallel_model, optimizer

