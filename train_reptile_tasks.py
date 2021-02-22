import open3d
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
torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)

from pointnet2.utils import pointnet2_utils

from gdn.utils import *
from gdn.detector.utils import *
from gdn import import_model_by_setting
import importlib
import copy
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to configuration file (in JSON)")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)

    if not os.path.exists(config['logdir']+'/ckpt'):
        os.makedirs(config['logdir']+'/ckpt')

    representation, dataset, my_collate_fn, base_model, model, optimizer, loss_function = import_model_by_setting(config)
    device = next(base_model.parameters()).device
    dataset.train()
    my_collate_fn.train()
    model.train()
    dataloader = DataLoader(dataset,
                            batch_size=config['task_size'],
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=False,
                            shuffle=True,
                            collate_fn=my_collate_fn)
    config = dataset.get_config()

    print('Num trainable params: %d'%count_parameters(base_model))

    epochs = config['epochs']
    start_epoch = 1
    logger = SummaryWriter(config['logdir'])
    pbar = tqdm(total=epochs*len(dataloader))
    json.dump(config, open(config['logdir']+'/settings.json', "w"))

    if 'pretrain_path' in config and os.path.exists(config['pretrain_path']):
        states = torch.load(config['pretrain_path'])
        base_model.load_state_dict(states['base_model'])
        loss_function.load_state_dict(states['loss_state'])
        optimizer.load_state_dict(states['optimizer_state'])
        start_epoch = states['epoch'] + 1
        pbar.set_description('Resume from last checkpoint... ')
        pbar.update((start_epoch-1)*len(dataloader))

    for e in range(start_epoch,1+epochs):
        loss_epoch = 0.0
        foreground_loss_epoch = 0.0
        l21_epoch = 0.0
        cls_loss_epoch = 0.0
        x_loss_epoch = 0.0
        y_loss_epoch = 0.0
        z_loss_epoch = 0.0
        rot_loss_epoch = 0.0
        uncert_epoch = 0.0
        n_iter = 0
        batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
        for _ in range(len(dataloader)):
            pc, volume, gt_poses = next(batch_iterator)
            if pc is None:
                batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
                pc, volume, gt_poses = next(batch_iterator)
            weights_before = copy.deepcopy(base_model.state_dict())
            for _ in range(config['innerepochs']):
                task_inds = np.random.permutation(len(pc))
                for task_i in task_inds:
                    batch_inds = np.random.permutation(len(pc[task_i]))
                    for start in range(0, len(pc[task_i]), config['batch_size']):
                        mbinds = batch_inds[start:start+config['batch_size']]
                        with torch.autograd.detect_anomaly():
                            optimizer.zero_grad()
                            pred, ind, att, l21 = model(pc[task_i][mbinds].cuda())
                            l21 = l21.mean()
                            (loss, foreground_loss, cls_loss,
                                x_loss, y_loss, z_loss,
                                rot_loss, ws, uncert) = loss_function(pred, ind, att, volume[task_i][mbinds].cuda())
                            loss += config['l21_reg_rate'] * l21 # l21 regularization (increase diversity)
                            loss.backward()
                            nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
                            optimizer.step()
                        n_iter += 1
                        loss_epoch += loss.item()
                        foreground_loss_epoch += foreground_loss
                        l21_epoch += l21.item()
                        cls_loss_epoch += cls_loss
                        x_loss_epoch += x_loss
                        y_loss_epoch += y_loss
                        z_loss_epoch += z_loss
                        rot_loss_epoch += rot_loss
                        uncert_epoch += uncert.item()
                        write_hwstat(config['logdir'])
            weights_after = base_model.state_dict()
            outerstepsize = config['outerstepsize0'] * (1.0 - (e-1.0) / epochs) # linear schedule
            base_model.load_state_dict({name : weights_before[name] + (weights_after[name] - weights_before[name]) * outerstepsize
                                        for name in weights_before})
            pbar.set_description('[%d/%d] iter: %d loss: %.2f reg: %.2f outer_lr: %.4e'%(e, epochs, n_iter, loss_epoch/n_iter, l21_epoch/n_iter, outerstepsize))
            pbar.update(1)

        loss_epoch /= n_iter
        foreground_loss_epoch /= n_iter
        l21_epoch /= n_iter
        cls_loss_epoch /= n_iter
        x_loss_epoch /= n_iter
        y_loss_epoch /= n_iter
        z_loss_epoch /= n_iter
        rot_loss_epoch /= n_iter
        uncert_epoch /= n_iter
        logger.add_scalar('train/loss', loss_epoch, e)
        logger.add_scalar('train/foreground_loss', foreground_loss_epoch, e)
        logger.add_scalar('train/sparisity', l21_epoch, e)
        logger.add_scalar('train/cls_loss', cls_loss_epoch, e)
        logger.add_scalar('train/x_loss', x_loss_epoch, e)
        logger.add_scalar('train/y_loss', y_loss_epoch, e)
        logger.add_scalar('train/z_loss', z_loss_epoch, e)
        logger.add_scalar('train/rot_loss', rot_loss_epoch, e)
        logger.add_scalar('train/uncert', uncert_epoch, e)

        logger.add_scalar('loss_weights/foreground', ws[0], e)
        logger.add_scalar('loss_weights/cls', ws[1], e)
        logger.add_scalar('loss_weights/x', ws[2], e)
        logger.add_scalar('loss_weights/y', ws[3], e)
        logger.add_scalar('loss_weights/z', ws[4], e)
        logger.add_scalar('loss_weights/rot', ws[5], e)

        torch.save({
            'base_model': base_model.state_dict(),
            'loss_state': loss_function.state_dict() if hasattr(loss_function, 'state_dict') else None,
            'optimizer_state': optimizer.state_dict(),
            'epoch': e,
            }, config['logdir']+'/ckpt/w-%d.pth'%e)

    logger.close()
    pbar.close()
