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

import higher
from pointnet2.utils import pointnet2_utils

from gdn.utils import *
from gdn.detector.utils import *
from gdn import import_model_by_setting
import importlib
from argparse import ArgumentParser

def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad

def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()

@torch.enable_grad()
def hv_prod(in_grad, x, params, lamb=10.0):
   hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
   hv = torch.nn.utils.parameters_to_vector(hv).detach()
   # precondition with identity matrix
   return hv/lamb + x

def init_vec_to_grad(model):
    param_dtypes = [ (param.numel(), param.size()) for param in model.parameters() ]
    def vec_to_grad(vec):
       pointer = 0
       res = []
       for (num_param, shape) in param_dtypes:
           res.append(vec[pointer:pointer+num_param].view(*shape).data)
           pointer += num_param
       return res
    return vec_to_grad

@torch.no_grad()
def cg(in_grad, outer_grad, params, vec_to_grad, n_cg=1):
    x = outer_grad.clone().detach()
    r = outer_grad.clone().detach() - hv_prod(in_grad, x, params)
    p = r.clone().detach()
    for i in range(n_cg):
        Ap = hv_prod(in_grad, p, params)
        alpha = (r @ r)/(p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new)/(r @ r)
        p = r_new + beta * p
        r = r_new.clone().detach()
    return vec_to_grad(x)

class TrainWrapper(nn.Module):
    def __init__(self, model, loss):
        super(TrainWrapper, self).__init__()
        self.model = model
        self.loss = loss
    def forward(self, x, y):
        pred, ind, att, l21 = self.model(x)
        return self.loss(pred, ind, att, y), l21

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

    representation, dataset, my_collate_fn, base_model, _, _, loss_function = import_model_by_setting(config)
    model = base_model

    model_with_loss = TrainWrapper(model, loss_function)
    inner_optimizer = optim.SGD(model_with_loss.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    outer_optimizer = optim.Adam(model_with_loss.parameters(), lr=config['outerstepsize0'])
    vec_to_grad = init_vec_to_grad(model_with_loss)

    dataset.train()
    my_collate_fn.train()
    model_with_loss.train()
    dataloader = DataLoader(dataset,
                            batch_size=config['task_batch_size'],
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

    len_meta_set = config['batch_size_meta_train']+config['batch_size_meta_test']
    len_meta_train = config['batch_size_meta_train']
    len_meta_test = config['batch_size_meta_test']

    if 'pretrain_path' in config and os.path.exists(config['pretrain_path']):
        states = torch.load(config['pretrain_path'])
        base_model.load_state_dict(states['base_model'])
        loss_function.load_state_dict(states['loss_state'])
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
            # Task sampling order (random)
            task_inds = np.random.permutation(len(pc))
            for task_i in task_inds:
                # Sample a task_i
                batch_inds = np.random.permutation(len(pc[task_i]))
                meta_test_ind = batch_inds[:len_meta_test]
                meta_train_ind = batch_inds[len_meta_test:len_meta_set]
                assert len(meta_train_ind) > 0
                assert len(meta_test_ind) > 0
                model_with_loss.zero_grad()
                grad_list = []
                with higher.innerloop_ctx(model_with_loss, inner_optimizer, track_higher_grads=False) as (fmodel, diffopt):
                    for t in range(config['innerepochs']+1):
                        if t > 0:
                            diffopt.step(in_loss)
                        (in_loss, foreground_loss, cls_loss,
                            x_loss, y_loss, z_loss,
                            rot_loss, ws, uncert), l21 = fmodel(pc[task_i][meta_train_ind].cuda(), volume[task_i][meta_train_ind].cuda())
                        l21 = l21.mean()
                        in_loss += config['l21_reg_rate'] * l21 # l21 regularization (increase diversity)
                    (outer_loss, foreground_loss, cls_loss,
                       x_loss, y_loss, z_loss,
                       rot_loss, ws, uncert), l21 = fmodel(pc[task_i][meta_test_ind].cuda(), volume[task_i][meta_test_ind].cuda())
                    l21 = l21.mean()
                    outer_loss += config['l21_reg_rate'] * l21 # l21 regularization (increase diversity)
                    loss = outer_loss.detach()
                    params = list(fmodel.parameters(time=-1))
                    in_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(in_loss, params, create_graph=True))
                    outer_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(outer_loss, params))
                    implicit_grad = cg(in_grad, outer_grad, params, vec_to_grad, n_cg=2)
                    grad_list.append(implicit_grad)

                outer_optimizer.zero_grad()
                weight = torch.ones(len(grad_list))
                weight = weight / torch.sum(weight)
                grad = mix_grad(grad_list, weight)
                grad_log = apply_grad(model, grad)
                outer_optimizer.step()

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
            pbar.set_description('[%d/%d] iter: %d loss: %.2f reg: %.2f'%(e, epochs, n_iter, loss_epoch/n_iter, l21_epoch/n_iter))
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
            'epoch': e,
            }, config['logdir']+'/ckpt/w-%d.pth'%e)

    logger.close()
    pbar.close()
