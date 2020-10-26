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

from gdn.utils import *
from gdn.detector.utils import *
from gdn import import_model_by_setting
import importlib
import copy
from argparse import ArgumentParser
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=3)

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
    dataset.train()
    my_collate_fn.train()
    dataloader = DataLoaderX(dataset,
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
        n_iter = 0
        model.train()
        my_collate_fn.train()
        dataset.train()
        batch_iterator = iter(dataloader)
        for _ in range(len(dataloader)):
            try:
                pc, volume, gt_poses, mask = next(batch_iterator)
            except StopIteration:
                warnings.warn('Oops! Something not right. Please check Ur Dataloader or buy more RAM :|')
                batch_iterator = iter(dataloader)
                pc, volume, gt_poses, mask = next(batch_iterator)
            optimizer.zero_grad()

            pred = model(pc.cuda())
            loss = loss_function(pred, volume.cuda(), mask.cuda())
            loss.backward()
            optimizer.step()
            n_iter += 1
            loss_epoch += loss.item()
            pbar.set_description('[%d/%d][%d/%d]: loss: %.4f '%(e, epochs, n_iter, len(dataloader), loss.item()))
            pbar.update(1)
            write_hwstat(config['logdir'])

        loss_epoch /= n_iter
        logger.add_scalar('train/loss', loss_epoch, e)

        if e % config['eval_freq'] == 0:
            loss_epoch = 0.0
            n_iter = 0
            model.eval()
            my_collate_fn.eval()
            dataset.eval()
            batch_iterator = iter(dataloader)
            with torch.no_grad():
                for _ in tqdm(range(len(dataloader))):
                    try:
                        pc, volume, gt_poses, mask = next(batch_iterator)
                    except StopIteration:
                        warnings.warn('Oops! Something not right. Please check Ur Dataloader or buy more RAM :|')
                        batch_iterator = iter(dataloader)
                        pc, volume, gt_poses, mask = next(batch_iterator)
                    pred = model(pc.cuda())
                    loss = loss_function(pred, volume.cuda(), mask.cuda())
                    n_iter += 1
                    loss_epoch += loss.item()
                    write_hwstat(config['logdir'])

                loss_epoch /= n_iter
                logger.add_scalar('eval/loss', loss_epoch, e)

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
