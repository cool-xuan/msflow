import os
import math
import datetime

import numpy as np
import torch

def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def save_weights(epoch, parallel_flows, fusion_flow, model_name, ckpt_dir, optimizer=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    file_name = '{}.pt'.format(model_name)
    file_path = os.path.join(ckpt_dir, file_name)
    print('Saving weights to {}'.format(file_path))
    state = {'epoch': epoch,
             'fusion_flow': fusion_flow.state_dict(),
             'parallel_flows': [parallel_flow.state_dict() for parallel_flow in parallel_flows]}
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    torch.save(state, file_path)


def load_weights(parallel_flows, fusion_flow, ckpt_path, optimizer=None):
    print('Loading weights from {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    
    fusion_state = state_dict['fusion_flow']
    maps = {}
    for i in range(len(parallel_flows)):
        maps[fusion_flow.module_list[i].perm.shape[0]] = i
    temp = dict()
    for k, v in fusion_state.items():
        if 'perm' not in k:
            continue
        temp[k.replace(k.split('.')[1], str(maps[v.shape[0]]))] = v
    for k, v in temp.items():
        fusion_state[k] = v
    fusion_flow.load_state_dict(fusion_state, strict=False)

    for parallel_flow, state in zip(parallel_flows, state_dict['parallel_flows']):
        parallel_flow.load_state_dict(state, strict=False)

    if optimizer:
        optimizer.load_state_dict(state_dict['optimizer'])

    return state_dict['epoch']

class Score_Observer:
    def __init__(self, name, total_epochs):
        self.name = name
        self.total_epochs = total_epochs
        self.max_epoch = 0
        self.max_score = 0.0
        self.last_score = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        best = False
        if score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            best = True
        if print_score:
            self.print_score(epoch)
        
        return best

    def print_score(self, epoch):
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"), 
            'Epoch [{:d}/{:d}] {:s}: last: {:.2f}\tmax: {:.2f}\tepoch_max: {:d}'.format(
                epoch, self.total_epochs-1, self.name, self.last, self.max_score, self.max_epoch))

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None