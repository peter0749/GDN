import os
import sys
import json
import warnings
from tensorboardX import SummaryWriter
from tqdm import tqdm

import numba as nb
warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True

from model.utils import *
from model.detector.utils import *
from model import import_model_by_setting
import importlib


def import_config_trick(path):
    # Kind of hack. TODO: Use a YAML to manage configuration
    directory, basename = os.path.split(path)
    sys.path.insert(0, directory)
    pyfilename = basename[:-3] # -.py
    config_parent = importlib.import_module(pyfilename)
    del sys.path[0]
    return config_parent.config


if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    #mp.set_start_method('forkserver', force=True)

    config = import_config_trick(sys.argv[1])

    if not os.path.exists(config['logdir']+'/ckpt'):
        os.makedirs(config['logdir']+'/ckpt')

    representation, dataset, my_collate_fn, base_model, model, optimizer = import_model_by_setting(config)
    my_collate_fn.train()
    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=True,
                            shuffle=True,
                            collate_fn=my_collate_fn)

    print('Num trainable params: %d'%count_parameters(base_model))

    epochs = config['epochs']
    start_epoch = 1
    logger = SummaryWriter(config['logdir'])
    pbar = tqdm(total=epochs*len(dataloader))
    json.dump(config, open(config['logdir']+'/settings.json', "w"))

    best_tpr2 = -1.0

    if 'pretrain_path' in config and os.path.exists(config['pretrain_path']):
        states = torch.load(config['pretrain_path'])
        base_model.load_state_dict(states['base_model'])
        if hasattr(representation, 'loss_object') and 'loss_state' in states and (not states['loss_state'] is None):
            representation.loss_object.load_state_dict(states['loss_state'])
        best_tpr2 = states['best_tpr2']
        optimizer.load_state_dict(states['optimizer_state'])
        start_epoch = states['epoch'] + 1
        pbar.set_description('Resume from last checkpoint... ')
        pbar.update((start_epoch-1)*len(dataloader))

    for e in range(start_epoch,1+epochs):
        loss_epoch = 0.0
        cls_loss_epoch = 0.0
        x_loss_epoch = 0.0
        y_loss_epoch = 0.0
        z_loss_epoch = 0.0
        rot_loss_epoch = 0.0
        uncert_epoch = 0.0
        n_iter = 0
        model.train()
        my_collate_fn.train()
        dataset.train()
        batch_iterator = iter(dataloader)
        for _ in range(len(dataloader)):
            try:
                pc, volume, indices, reverse_lookup_index, gt_poses = next(batch_iterator)
            except StopIteration:
                warnings.warn('Oops! Something not right. Please check Ur Dataloader or buy more RAM :|')
                batch_iterator = iter(dataloader)
                pc, volume, indices, reverse_lookup_index, gt_poses = next(batch_iterator)
            optimizer.zero_grad()

            pred = model(pc.cuda(), [pt_idx.cuda() for pt_idx in indices])
            (loss, cls_loss,
                x_loss, y_loss, z_loss,
                rot_loss, ws, uncert) = representation.compute_loss(pred, volume.cuda())
            loss.backward()
            optimizer.step()
            n_iter += 1
            loss_epoch += loss.item()
            cls_loss_epoch += cls_loss
            x_loss_epoch += x_loss
            y_loss_epoch += y_loss
            z_loss_epoch += z_loss
            rot_loss_epoch += rot_loss
            uncert_epoch += uncert.item()
            pbar.set_description('[%d/%d][%d/%d]: loss: %.2f '%(e, epochs, n_iter, len(dataloader), loss.item()))
            pbar.update(1)
            write_hwstat(config['logdir'])

        loss_epoch /= n_iter
        cls_loss_epoch /= n_iter
        x_loss_epoch /= n_iter
        y_loss_epoch /= n_iter
        z_loss_epoch /= n_iter
        rot_loss_epoch /= n_iter
        uncert_epoch /= n_iter
        logger.add_scalar('train/loss', loss_epoch, e)
        logger.add_scalar('train/cls_loss', cls_loss_epoch, e)
        logger.add_scalar('train/x_loss', x_loss_epoch, e)
        logger.add_scalar('train/y_loss', y_loss_epoch, e)
        logger.add_scalar('train/z_loss', z_loss_epoch, e)
        logger.add_scalar('train/rot_loss', rot_loss_epoch, e)
        logger.add_scalar('train/uncert', uncert_epoch, e)

        logger.add_scalar('loss_weights/cls', ws[0], e)
        logger.add_scalar('loss_weights/x', ws[1], e)
        logger.add_scalar('loss_weights/y', ws[2], e)
        logger.add_scalar('loss_weights/z', ws[3], e)
        logger.add_scalar('loss_weights/rot', ws[4], e)

        if e % config['eval_freq'] == 0:
            loss_epoch = 0.0
            cls_loss_epoch = 0.0
            x_loss_epoch = 0.0
            y_loss_epoch = 0.0
            z_loss_epoch = 0.0
            rot_loss_epoch = 0.0
            tpr = 0.0
            fpr = 0.0
            fnr = 0.0
            spr = 0.0
            tpr_2 = 0.0
            n_pos = 0.0
            n_gt  = 0.0
            n_iter = 0
            model.eval()
            my_collate_fn.eval()
            dataset.eval()
            batch_iterator = iter(dataloader)
            with torch.no_grad():
                for _ in tqdm(range(len(dataloader))):
                    try:
                        pc, volume, indices, reverse_lookup_index, gt_poses = next(batch_iterator)
                    except StopIteration:
                        warnings.warn('Oops! Something not right. Please check Ur Dataloader or buy more RAM :|')
                        batch_iterator = iter(dataloader)
                        pc, volume, indices, reverse_lookup_index, gt_poses = next(batch_iterator)
                    pred = model(pc.cuda(), [pt_idx.cuda() for pt_idx in indices])
                    (loss, cls_loss,
                        x_loss, y_loss, z_loss,
                        rot_loss, ws, uncert) = representation.compute_loss(pred, volume.cuda())
                    n_iter += 1
                    loss_epoch += loss.item()
                    cls_loss_epoch += cls_loss
                    x_loss_epoch += x_loss
                    y_loss_epoch += y_loss
                    z_loss_epoch += z_loss
                    rot_loss_epoch += rot_loss
                    pred_poses = representation.retrive_from_feature_volume_batch(
                                pc.cpu().numpy(),
                                reverse_lookup_index,
                                pred.detach().cpu().numpy(),
                                n_output=config['eval_pred_maxbox'],
                                threshold=config['eval_pred_threshold'],
                                nms=False
                            )
                    tp, fp, fn, sp, n_p, n_t = batch_metrics(pred_poses, gt_poses, **config)
                    tpr += tp
                    fpr += fp
                    fnr += fn
                    spr += sp
                    tpr_2 += (tp + sp)
                    n_pos += n_p
                    n_gt  += n_t
                    write_hwstat(config['logdir'])
                spr = spr / max(1, tpr_2)
                tpr = tpr / max(1, n_pos)
                tpr_2 = tpr_2 / max(1, n_pos)
                fpr = fpr / max(1, n_pos)
                fnr = fnr / max(1, n_gt)
                logger.add_scalar('eval/tpr', tpr, e)
                logger.add_scalar('eval/tpr_2', tpr_2, e)
                logger.add_scalar('eval/spr', spr, e)
                logger.add_scalar('eval/fpr', fpr, e)
                logger.add_scalar('eval/fnr', fnr, e)
                logger.add_scalar('eval/f-1', 2 * tpr / (2 * tpr + fnr + fpr + 1e-8), e)
                logger.add_scalar('eval/f-0.5', (1+0.5**2) * tpr / ((1+0.5**2) * tpr + 0.5**2 * fnr + fpr + 1e-8), e)
                logger.add_scalar('eval/f-2', (1+2**2) * tpr / ((1+2**2) * tpr + 2**2 * fnr + fpr + 1e-8), e)

                loss_epoch /= n_iter
                cls_loss_epoch /= n_iter
                x_loss_epoch /= n_iter
                y_loss_epoch /= n_iter
                z_loss_epoch /= n_iter
                rot_loss_epoch /= n_iter
                logger.add_scalar('eval/loss', loss_epoch, e)
                logger.add_scalar('eval/cls_loss', cls_loss_epoch, e)
                logger.add_scalar('eval/x_loss', x_loss_epoch, e)
                logger.add_scalar('eval/y_loss', y_loss_epoch, e)
                logger.add_scalar('eval/z_loss', z_loss_epoch, e)
                logger.add_scalar('eval/rot_loss', rot_loss_epoch, e)

                if tpr_2 > best_tpr2:
                    best_tpr2 = tpr_2
                    torch.save({
                        'base_model': base_model.state_dict(),
                        'loss_state': representation.loss_object.state_dict() if hasattr(representation, 'loss_object') else None,
                        'tpr_2': tpr_2,
                        'best_tpr2': best_tpr2,
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': e,
                        }, config['logdir']+'/best.ckpt')
                torch.save({
                    'base_model': base_model.state_dict(),
                    'loss_state': representation.loss_object.state_dict() if hasattr(representation, 'loss_object') else None,
                    'tpr_2': tpr_2,
                    'best_tpr2': best_tpr2,
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': e,
                    }, config['logdir']+'/ckpt/w-%d.pth'%e)

    logger.close()
    pbar.close()


