# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable


def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    if 'allimg' in caption_model:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * Variable(reward) * Variable(mask)
        output = torch.sum(output) / torch.sum(mask)
        return output


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / (torch.sum(mask) + 1e-6)
        return output


class TAPModelCriterion(nn.Module):
    def __init__(self):
        super(TAPModelCriterion, self).__init__()

    def forward(self, scores, masks, labels, w1):
        """
        Uses weighted BCE to calculate loss
        """
        # masks, labels, w1 = [torch.Tensor(_) for _ in [masks, labels, w1]]
        # w1 = torch.FloatTensor(w1).type_as(outputs.data)
        w0 = 1. - w1
        labels = labels.mul(masks)
        weights = labels.mul(w0.expand(labels.size())) + (1. - labels).mul(w1.expand(labels.size()))
        weights = weights.view(-1)
        # labels = torch.autograd.Variable(labels.view(-1))
        # masks = torch.autograd.Variable(masks.view(-1))
        labels = labels.view(-1)
        masks = masks.view(-1)
        scores = scores.view(-1).mul(masks)
        criterion = torch.nn.BCELoss(weight=weights)
        loss = criterion(scores, labels) * w0.shape[0]
        return loss


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for i, param in enumerate(group['params']):
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def fix_model_parameters(model):
    for para in model.parameters():
        para.requires_grad = False


def unfix_model_parameters(model):
    for para in model.parameters():
        para.requires_grad = True

