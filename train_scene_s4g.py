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
    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=True,
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
        if 'loss_state' in states and (not states['loss_state'] is None) and hasattr(loss_function, 'load_state_dict'):
            loss_function.load_state_dict(states['loss_state'])
        optimizer.load_state_dict(states['optimizer_state'])
        start_epoch = states['epoch'] + 1
        pbar.set_description('Resume from last checkpoint... ')
        pbar.update((start_epoch-1)*len(dataloader))

    for e in range(start_epoch,1+epochs):
        loss_epoch = 0.0
        cls_loss_epoch = 0.0
        rot_loss_epoch = 0.0
        trans_loss_epoch = 0.0
        uncert_epoch = 0.0
        n_iter = 0
        model.train()
        my_collate_fn.train()
        dataset.train()
        batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
        for _ in range(len(dataloader)):
            pc, volume, gt_poses = next(batch_iterator)
            if pc is None:
                batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
                pc, volume, gt_poses = next(batch_iterator)
            pc = pc.cuda()
            volume = volume.cuda()
            optimizer.zero_grad()

            pred = model(pc)
            (loss, cls_loss, rot_loss, trans_loss, ws, uncert) = loss_function(pred, volume.cuda())
            loss.backward()
            optimizer.step()
            n_iter += 1
            loss_epoch += loss.item()
            cls_loss_epoch += cls_loss
            rot_loss_epoch += rot_loss
            trans_loss_epoch += trans_loss
            uncert_epoch += uncert.item()
            pbar.set_description('[%d/%d][%d/%d]: loss: %.2f'%(e, epochs, n_iter, len(dataloader), loss.item()))
            pbar.update(1)
            write_hwstat(config['logdir'])

        loss_epoch /= n_iter
        cls_loss_epoch /= n_iter
        rot_loss_epoch /= n_iter
        trans_loss_epoch /= n_iter
        uncert_epoch /= n_iter
        logger.add_scalar('train/loss', loss_epoch, e)
        logger.add_scalar('train/cls_loss', cls_loss_epoch, e)
        logger.add_scalar('train/rot_loss', rot_loss_epoch, e)
        logger.add_scalar('train/trans_loss', trans_loss_epoch, e)
        logger.add_scalar('train/uncert', uncert_epoch, e)

        logger.add_scalar('loss_weights/cls', ws[0], e)
        logger.add_scalar('loss_weights/rot', ws[1], e)
        logger.add_scalar('loss_weights/trans', ws[2], e)

        torch.save({
            'base_model': base_model.state_dict(),
            'loss_state': loss_function.state_dict() if hasattr(loss_function, 'state_dict') else None,
            'optimizer_state': optimizer.state_dict(),
            'epoch': e,
            }, config['logdir']+'/ckpt/w-%d.pth'%e)

    logger.close()
    pbar.close()
