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

def next_batch(iterator):
    try:
        pc, volume, mask, gt = next(iterator)
        if pc is None:
            return None, None, None, None
    except:
        return None, None, None, None
    pc = pc.cuda()
    volume = volume.cuda()
    mask = mask.cuda()
    return pc, volume, mask, gt

if __name__ == '__main__':
    # In[14]:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    with open(args.config, 'r') as fp:
        config = json.load(fp)

    if not os.path.exists(config['logdir']+'/ckpt'):
        os.makedirs(config['logdir']+'/ckpt')

    representation, dataset, my_collate_fn, base_model, _, optimizer, loss_function = import_model_by_setting(config)
    model = base_model

    dataset_query, dataset_support_in_eval, dataset_support = dataset
    dataset_query.train()
    dataset_support_in_eval.train() # always train mode
    dataset_support.train()
    my_collate_fn.train()
    dataloader_query = DataLoader(dataset_query,
                            batch_size=config['query_batch_size'],
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=False,
                            shuffle=True,
                            collate_fn=my_collate_fn)
    dataloader_support = DataLoader(dataset_support,
                            batch_size=config['support_batch_size'],
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=False,
                            shuffle=True,
                            collate_fn=my_collate_fn)
    dataloader_support_in_eval = DataLoader(dataset_support_in_eval,
                            batch_size=config['support_batch_size'],
                            num_workers=config['num_workers_dataloader'],
                            pin_memory=False,
                            shuffle=True,
                            collate_fn=my_collate_fn)
    config = dataset_query.get_config()

    print('Num trainable params: %d'%count_parameters(base_model))

    epochs = config['epochs']
    start_epoch = 1
    logger = SummaryWriter(config['logdir'])
    pbar = tqdm(total=epochs*len(dataloader_query))
    json.dump(config, open(config['logdir']+'/settings.json', "w"))

    best_tpr2 = -1.0

    if 'pretrain_path' in config and os.path.exists(config['pretrain_path']):
        states = torch.load(config['pretrain_path'])
        base_model.load_state_dict(states['base_model'])
        if 'loss_state' in states and (not states['loss_state'] is None) and hasattr(loss_function, 'load_state_dict'):
            loss_function.load_state_dict(states['loss_state'])
        best_tpr2 = states['best_tpr2']
        optimizer.load_state_dict(states['optimizer_state'])
        start_epoch = states['epoch'] + 1
        pbar.set_description('Resume from last checkpoint... ')
        pbar.update((start_epoch-1)*len(dataloader_query))

    for e in range(start_epoch,1+epochs):
        info_namespace = [
            "Train/Loss",
            "Train/DPP_regularization",
            "Train/Uncertainty",
            "Train/Loss_q",
            "Train/Foreground_q",
            "Train/Coarse_rotation_q",
            "Train/X_normalized_loss_q",
            "Train/Y_normalized_loss_q",
            "Train/Z_normalized_loss_q",
            "Train/Rotation_loss_q",
            "Train/Loss_s",
            "Train/Foreground_s",
            "Train/Coarse_rotation_s",
            "Train/X_normalized_loss_s",
            "Train/Y_normalized_loss_s",
            "Train/Z_normalized_loss_s",
            "Train/Rotation_loss_s",
            "Train/Loss_qq",
            "Train/Foreground_qq",
            "Train/Coarse_rotation_qq",
            "Train/X_normalized_loss_qq",
            "Train/Y_normalized_loss_qq",
            "Train/Z_normalized_loss_qq",
            "Train/Rotation_loss_qq",
            "Train/Loss_ss",
            "Train/Foreground_ss",
            "Train/Coarse_rotation_ss",
            "Train/X_normalized_loss_ss",
            "Train/Y_normalized_loss_ss",
            "Train/Z_normalized_loss_ss",
            "Train/Rotation_loss_ss",
        ]
        info = np.zeros(len(info_namespace)+1, dtype=np.float32)
        model.train()
        my_collate_fn.train()
        dataset_query.train()
        dataset_support.train()
        batch_iterator_query = iter(dataloader_query)
        batch_iterator_support = iter(dataloader_support)
        for _ in range(len(dataloader_query)):
            # Load query data:
            pc_query, volume_query, mask_query, gt_query = next_batch(batch_iterator_query)
            if pc_query is None:
                batch_iterator_query = iter(dataloader_query)
                pc_query, volume_query, mask_query, gt_query = next_batch(batch_iterator_query)

            # Load support data:
            pc_support, volume_support, mask_support, gt_support = next_batch(batch_iterator_support)
            if pc_support is None:
                batch_iterator_support = iter(dataloader_support)
                pc_support, volume_support, mask_support, gt_support = next_batch(batch_iterator_support)

            optimizer.zero_grad()

            # pseudo-label probagation: support -> query
            pred_q, ind_q, att_q, prototype_s, l21_q = model(pc_support, mask_support, pc_query, None)
            # pseudo-label probagation: query -> support
            pred_s, ind_s, att_s, prototype_q, l21_s = model(pc_query, mask_query, pc_support, None)
            # self consistency: support -> support
            pred_ss, ind_ss, att_ss = model(None, None, pc_support, prototype_s)[:3]
            # self consistency: query -> query
            pred_qq, ind_qq, att_qq = model(None, None, pc_query, prototype_q)[:3]

            l21 = torch.cat((l21_q.view(-1), l21_s.view(-1))).mean()
            # Cross-Domain
            (loss_q, foreground_loss_q, cls_loss_q,
                x_loss_q, y_loss_q, z_loss_q,
                rot_loss_q, ws, uncert) = loss_function(pred_q, ind_q, att_q, volume_query)
            (loss_s, foreground_loss_s, cls_loss_s,
                x_loss_s, y_loss_s, z_loss_s,
                rot_loss_s, _, _) = loss_function(pred_s, ind_s, att_s, volume_support)
            # In-Domain (self-consistency)
            (loss_qq, foreground_loss_qq, cls_loss_qq,
                x_loss_qq, y_loss_qq, z_loss_qq,
                rot_loss_qq, _, _) = loss_function(pred_qq, ind_qq, att_qq, volume_query)
            (loss_ss, foreground_loss_ss, cls_loss_ss,
                x_loss_ss, y_loss_ss, z_loss_ss,
                rot_loss_ss, _, _) = loss_function(pred_ss, ind_ss, att_ss, volume_support)
            loss = loss_q + loss_s + loss_qq + loss_ss + config['l21_reg_rate'] * l21
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0, norm_type=2)
            optimizer.step()
            info += np.array([
                    loss.item(),
                    l21.item(),
                    uncert.item(),
                    loss_q.item(),
                    foreground_loss_q,
                    cls_loss_q,
                    x_loss_q,
                    y_loss_q,
                    z_loss_q,
                    rot_loss_q,
                    loss_s.item(),
                    foreground_loss_s,
                    cls_loss_s,
                    x_loss_s,
                    y_loss_s,
                    z_loss_s,
                    rot_loss_s,
                    loss_qq.item(),
                    foreground_loss_qq,
                    cls_loss_qq,
                    x_loss_qq,
                    y_loss_qq,
                    z_loss_qq,
                    rot_loss_qq,
                    loss_ss.item(),
                    foreground_loss_ss,
                    cls_loss_ss,
                    x_loss_ss,
                    y_loss_ss,
                    z_loss_ss,
                    rot_loss_ss,
                    1
                ],dtype=np.float32)
            pbar.set_description('[%d/%d][%d/%d]: loss_s: %.2f loss_q: %.2f loss_ss: %.2f loss_qq: %.2f reg: %.2f'%(e, epochs, info[-1], len(dataloader_query), loss_s.item(), loss_q.item(), loss_ss.item(), loss_qq.item(), l21.item()))
            pbar.update(1)
            write_hwstat(config['logdir'])

        info /= info[-1]
        for i in range(len(info_namespace)):
            logger.add_scalar(info_namespace[i], info[i], e)

        if e % config['eval_freq'] == 0:
            info = np.zeros(11, dtype=np.float32)
            info_namespace = [
                "Valid/Loss_q",
                "Valid/Foreground_q",
                "Valid/Coarse_rotation_q",
                "Valid/X_normalized_loss_q",
                "Valid/Y_normalized_loss_q",
                "Valid/Z_normalized_loss_q",
                "Valid/Rotation_loss_q",
                "Valid/mAP_q",
                "Valid/TPR_q",
                ] # 9
            # n_pos
            # n_iter
            model.eval()
            my_collate_fn.eval()
            dataset_query.eval()
            dataset_support_in_eval.train() # Use GT in training data in same domain as support

            with torch.no_grad():
                batch_iterator_query = iter(dataloader_query)
                batch_iterator_support = iter(dataloader_support_in_eval)
                for _ in tqdm(range(len(dataloader_query))):
                    # Load query data:
                    pc_query, volume_query, _, gt_poses = next(batch_iterator_query)
                    if pc_query is None:
                        batch_iterator_query = iter(dataloader_query)
                        pc_query, volume_query, _, gt_poses = next(batch_iterator_query)
                    pc_query = pc_query.cuda()
                    volume_query = volume_query.cuda()

                    # Load support data:
                    pc_support, volume_support, support_mask, _ = next(batch_iterator_support)
                    if pc_support is None:
                        batch_iterator_support = iter(dataloader_support_in_eval)
                        pc_support, volume_support, support_mask, _ = next(batch_iterator_support)
                    pc_support = pc_support.cuda()
                    support_mask = support_mask.cuda()
                    #volume_support = volume_support.cuda()

                    pred, ind, att = model(pc_support, support_mask, pc_query, None)[:3]
                    (loss, foreground_loss, cls_loss,
                        x_loss, y_loss, z_loss,
                        rot_loss, _, _) = loss_function(pred, ind, att, volume_query)
                    pc_subsampled = pointnet2_utils.gather_operation(pc_query.transpose(1, 2).contiguous(), ind)
                    pc_subsampled = pc_subsampled.transpose(1, 2).cpu().numpy()
                    pred_poses = representation.retrive_from_feature_volume_batch(
                                pc_subsampled,
                                pred.detach().cpu().numpy(),
                                n_output=config['eval_pred_maxbox'],
                                threshold=config['eval_pred_threshold'],
                                nms=False
                            )
                    tp, fp, fn, sp, n_p, n_t, mAP = batch_metrics(pred_poses, gt_poses, **config)
                    info += np.array([
                            loss.item(),
                            foreground_loss,
                            cls_loss,
                            x_loss,
                            y_loss,
                            z_loss,
                            rot_loss,
                            mAP,
                            tp+sp,
                            n_p,
                            1
                        ], dtype=np.float32)
                    write_hwstat(config['logdir'])
                info[:8] /= info[-1]
                info[8] /= info[-2]
                mean_mAP = info[7]
                tpr_2 = info[8]

                for i in range(len(info_namespace)):
                    logger.add_scalar(info_namespace[i], info[i], e)

                if tpr_2 > best_tpr2:
                    best_tpr2 = tpr_2
                    torch.save({
                        'base_model': base_model.state_dict(),
                        'loss_state': loss_function.state_dict() if hasattr(loss_function, 'state_dict') else None,
                        'tpr_2': tpr_2,
                        'mAP': mean_mAP,
                        'best_tpr2': best_tpr2,
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': e,
                        }, config['logdir']+'/best.ckpt')
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
