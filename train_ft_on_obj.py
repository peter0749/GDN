import open3d

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
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

'''
class data_prefetcher(object):
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device=device)
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_gt = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_gt = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(self.device, non_blocking=True)
            #self.next_gt = self.next_gt

    def next(self):
        torch.cuda.default_stream(device=self.device).wait_stream(stream=self.stream)
        inputs = self.next_input
        target = self.next_target
        gt = self.next_gt
        self.preload()
        return inputs, target, gt
'''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to configuration file (in JSON)")
    parser.add_argument("obj", type=str, help="Path to configuration file (in JSON)")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)
        config["train_data"] += '/' + args.obj
        config["train_label"] += '/' + args.obj
        config["val_data"] = config["train_data"]
        config["val_label"] = config["train_label"]
        config["logdir"] += '_' + args.obj

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

    best_tpr2 = -1.0
    tpr_2 = -1.0
    mean_mAP = -1.0

    if 'pretrain_path' in config and os.path.exists(config['pretrain_path']):
        states = torch.load(config['pretrain_path'])
        base_model.load_state_dict(states['base_model'])
        if 'loss_state' in states and (not states['loss_state'] is None) and hasattr(loss_function, 'load_state_dict'):
            loss_function.load_state_dict(states['loss_state'])
        #best_tpr2 = states['best_tpr2']
        optimizer.load_state_dict(states['optimizer_state'])
        #start_epoch = states['epoch'] + 1
        pbar.set_description('Resume from last checkpoint... ')
        #pbar.update((start_epoch-1)*len(dataloader))

    for e in range(start_epoch,1+epochs):
        loss_epoch = 0.0
        foreground_loss_epoch = 0.0
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
        batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
        for _ in range(len(dataloader)):
            pc, volume, gt_poses = next(batch_iterator)
            if pc is None:
                batch_iterator = iter(dataloader) # data_prefetcher(dataloader, device)
                pc, volume, gt_poses = next(batch_iterator)
            pc = pc.cuda()
            volume = volume.cuda()
            optimizer.zero_grad()

            pred, ind, att = model(pc)[:3]
            (loss, foreground_loss, cls_loss,
                x_loss, y_loss, z_loss,
                rot_loss, ws, uncert) = loss_function(pred, ind, att, volume)
            loss.backward()
            optimizer.step()
            n_iter += 1
            loss_epoch += loss.item()
            foreground_loss_epoch += foreground_loss
            cls_loss_epoch += cls_loss
            x_loss_epoch += x_loss
            y_loss_epoch += y_loss
            z_loss_epoch += z_loss
            rot_loss_epoch += rot_loss
            uncert_epoch += uncert.item()
            pbar.set_description('[%d/%d][%d/%d]: loss: %.2f'%(e, epochs, n_iter, len(dataloader), loss.item()))
            pbar.update(1)
            write_hwstat(config['logdir'])

        loss_epoch /= n_iter
        foreground_loss_epoch /= n_iter
        cls_loss_epoch /= n_iter
        x_loss_epoch /= n_iter
        y_loss_epoch /= n_iter
        z_loss_epoch /= n_iter
        rot_loss_epoch /= n_iter
        uncert_epoch /= n_iter
        logger.add_scalar('train/loss', loss_epoch, e)
        logger.add_scalar('train/foreground_loss', foreground_loss_epoch, e)
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
            'tpr_2': tpr_2,
            'mAP': mean_mAP,
            'best_tpr2': best_tpr2,
            'optimizer_state': optimizer.state_dict(),
            'epoch': e,
            }, config['logdir']+'/ckpt/w-%d.pth'%e)

    logger.close()
    pbar.close()
