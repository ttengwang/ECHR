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


class ATTnormCriterion(nn.Module):
    def __init__(self):
        super(ATTnormCriterion, self).__init__()

    def forward(self, att_weights, att_mask, cg_mask, actionness, opt=
    {'diff':False,
     'crit_type':'mse',
     'data_type':'dot'}):

        '''
        :param att_weights: [proposal_num, seq_len, att_size]
        :param att_mask: [proposal_num, att_size]
        :param cg_mask: [proposal_num, seq_len]
        :param actionness: [proposal_num， att_size]
        :return:
        '''
        # truncate to the same size

        diff, data_type, crit_type = opt['diff'], opt['data_type'], opt['crit_type']

        attweight, attdot = att_weights
        if data_type== 'dot':
            weight = attdot
        elif data_type== 'rescale':
            weight = attweight

        cg_mask = cg_mask[:, :weight.size(1)]
        max_len = att_mask.shape[1]
        assert actionness[:,max_len:].sum().item()==0
        _att_weights = (weight * cg_mask.unsqueeze(2)).sum(1) / (1e-3 + cg_mask.sum(1, keepdim=True)) #[proposal_num, att_size]
        actionness = actionness[:,:max_len]

        label = (1 - actionness)  # [proposal_num, K+1]
        if crit_type == 'mse':
            positive = torch.norm(actionness * att_mask * _att_weights, p=2, dim=1)/torch.sqrt(1e-3+ (actionness * att_mask).sum(1))
            negative = torch.norm(label * att_mask *  _att_weights, p=2, dim=1) / torch.sqrt(1e-3+(label * att_mask).sum(1))

        elif crit_type == 'ce':
            inverse_weight = 1 - _att_weights * att_mask
            negative = (-label * torch.log(inverse_weight)).sum(1) / (label.sum(1) + 1e-3)

        if diff:
            loss = -(positive - negative).sum() / (cg_mask.sum(1)>0).sum()
        else:
            loss = negative.sum() / (cg_mask.sum(1)>0).sum()
        pdb.set_trace()
        return loss


# class ATTnormCriterion_CE(nn.Module):
#     def __init__(self):
#         super(ATTnormCriterion_CE, self).__init__()
#         self.crit = torch.nn.CrossEntropyLoss()
#
#     def forward(self, att_weights, att_mask, cg_mask, actionness):
#
#         '''
#         :param att_weights: [proposal_num, seq_len, att_size]
#         :param att_mask: [proposal_num, att_size]
#         :param cg_mask: [proposal_num, seq_len]
#         :param actionness: [proposal_num， att_size]
#         :return:
#         '''
#         # truncate to the same size
#
#         cg_mask = cg_mask[:, :att_weights.size(1)]
#         max_len = att_mask.shape[1]
#         assert actionness[:,max_len:].sum().item()==0
#         att_weights = (att_weights * cg_mask.unsqueeze(2)).sum(1) / cg_mask.sum(1) #[proposal_num, att_size]
#         actionness = actionness[:,:max_len]
#
#         masked_weight = att_weights * att_mask
#         label = 1 - actionness # [proposal_num, K+1]
#         inverse_weight = 1 - masked_weight
#         loss = (-label * torch.log(inverse_weight)).sum() / (label.sum() + 1e-3)
#
#         # loss = torch.norm(label * masked_weight, 2) / label.sum()
#         return loss

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

class TAPModelCriterion_2(nn.Module):
    def __init__(self):
        super(TAPModelCriterion_2, self).__init__()

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
        loss = criterion(scores, labels)* 128
        return loss

class DiffTAPModelCriterion(nn.Module):
    def __init__(self):
        super(DiffTAPModelCriterion, self).__init__()

    def forward(self, scores, masks, labels, event_boundary, w1, bw1):
        """
        Uses weighted BCE to calculate loss
        """
        score1 = scores[0]
        score2 = scores[1]
        w0 = 1. - w1
        labels = labels.mul(masks)
        weights = labels.mul(w0.expand(labels.size())) + (1. - labels).mul(w1.expand(labels.size()))
        weights = weights.view(-1)
        labels = labels.view(-1)
        masks = masks.view(-1)
        score1 = score1.view(-1).mul(masks)
        criterion = torch.nn.BCELoss(weight=weights)
        loss1 = criterion(score1, labels)

        bw0 = 1. - bw1

        bweights = event_boundary.mul(bw0.expand(event_boundary.size())) + (1. - event_boundary).mul(bw1.expand(event_boundary.size()))
        bweights = bweights.view(-1)
        event_boundary = event_boundary.view(-1)

        score2 = score2.view(-1)

        criterion2 = torch.nn.BCELoss(weight=bweights)
        loss2 = criterion2(score2, event_boundary)

        loss = loss1 + loss2
        return loss,loss1,loss2

class DesTAPModelCriterion(nn.Module):
    def __init__(self):
        super(DesTAPModelCriterion, self).__init__()

    def forward(self, scores, masks, labels, event_des_score, w1):
        """
        Uses weighted BCE to calculate loss
        """
        score1 = scores[0]
        score2 = scores[1]
        w0 = 1. - w1
        labels = labels.mul(masks)
        weights = labels.mul(w0.expand(labels.size())) + (1. - labels).mul(w1.expand(labels.size()))
        weights = weights.view(-1)
        labels = labels.view(-1)
        masks = masks.view(-1)
        score1 = score1.view(-1).mul(masks)
        criterion = torch.nn.BCELoss(weight=weights)
        loss1 = criterion(score1, labels)

        criterion2 = torch.nn.MSELoss()
        event_des_score = event_des_score.view(-1)
        score2 = score2.view(-1)
        loss2 = criterion2(score2, event_des_score)
        loss = loss1 + loss2
        return loss,loss1,loss2

class TAPModelCriterion_reverse(nn.Module):
    def __init__(self):
        super(TAPModelCriterion_reverse, self).__init__()

    def forward(self, outputs, masks, labels, w1):
        """
        Uses weighted BCE to calculate loss
        """
        # masks, labels, w1 = [torch.Tensor(_) for _ in [masks, labels, w1]]
        # w1 = torch.FloatTensor(w1).type_as(outputs.data)
        w0 = w1
        w1 = 1. - w1
        # w0 = 1 - w1
        labels = labels.mul(masks)
        weights = labels.mul(w0.expand(labels.size())) + (1. - labels).mul(w1.expand(labels.size()))
        weights = weights.view(-1)
        # labels = torch.autograd.Variable(labels.view(-1))
        # masks = torch.autograd.Variable(masks.view(-1))
        labels = labels.view(-1)
        masks = masks.view(-1)
        outputs = outputs.view(-1).mul(masks)
        criterion = torch.nn.BCELoss(weight=weights)
        loss = criterion(outputs, labels) * w0.shape[1]
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


def get_dummy_data(tap_opt, lm_opt):
    import numpy as np
    nwindows = tap_opt.batch_size
    num_of_proposals = 5
    max_length = 18
    w1 = 0.2
    data_ = {}
    video_length = np.random.randint(125, 140)
    caption_num = np.random.randint(1, 7)
    data_['fc_feats'] = np.random.rand(nwindows, video_length, tap_opt.video_dim).astype('float32')
    data_['att_feats'] = np.random.rand(nwindows, video_length, tap_opt.video_dim).astype('float32')
    data_['tap_labels'] = np.random.rand(nwindows, video_length, tap_opt.K).astype('float32')
    data_['tap_masks_for_loss'] = np.ones([nwindows, video_length, tap_opt.K]).astype('float32')
    data_['high_iou_mask'] = np.random.randint(0, 2, [tap_opt.batch_size, video_length, tap_opt.K]).astype('int')
    data_['high_iou_index'] = np.random.randint(1, caption_num + 1,
                                                [tap_opt.batch_size, video_length, tap_opt.K]).astype(
        'int') * data_['high_iou_mask'] - 1
    data_['proposal_num'] = proposal_num = data_['high_iou_mask'].sum().tolist()
    data_['lm_labels'] = np.random.randint(0, 2, [tap_opt.batch_size, caption_num, max_length]).astype('int')
    data_['lm_masks'] = np.ones([tap_opt.batch_size, caption_num, max_length]).astype('float32')
    data_['w1'] = np.array([w1]).astype('float32')
    data_['bounds'] = {}
    data_['bounds']['wrapped'] = 1

    for key, item in data_.items():
        if type(item) == np.ndarray and item.ndim > 2:
            data_[key] = item.squeeze(0)

    tap_indices_selecting_list = []
    lm_indices_selecting_list = []

    for i, row in enumerate(data_['high_iou_index']):
        for index in row:
            if not index == -1:
                tap_indices_selecting_list.append(i)
                lm_indices_selecting_list.append(index)
    shuffle_list = np.array(range(proposal_num))
    np.random.shuffle(shuffle_list)
    shuffle_list = shuffle_list[:100]
    data_['tap_ind_select_list'] = np.array(tap_indices_selecting_list)[shuffle_list]
    data_['lm_labels'] = data_['lm_labels'][lm_indices_selecting_list][shuffle_list]
    data_['lm_masks'] = data_['lm_masks'][lm_indices_selecting_list][shuffle_list]

    return data_
