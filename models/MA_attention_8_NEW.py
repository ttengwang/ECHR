# coding:utf-8
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class MA_Attention8(nn.Module):
    def __init__(self, opt):
        super(MA_Attention8, self).__init__()

        if 'ER1' in opt.event_context_type:
            opt.TSRM_input_dim =opt.video_dim
        elif 'ER2' in opt.event_context_type:
            opt.TSRM_input_dim =opt.hidden_dim
        elif 'ER3' in opt.event_context_type:
            opt.TSRM_input_dim = opt.video_dim + opt.hidden_dim
        else:
            assert False , 'feature_type wrong'

        opt.d_pos_vec = opt.d_feats
        self.h2a_layer = torch.nn.Linear(10, 10)  # hidden_dim: hidden state size of proposal LSTM

        self.output_dim = opt.d_o
        self.use_posit = opt.use_posit

        self.d_pos_vec = opt.d_pos_vec
        self.event_emb = nn.Linear(opt.TSRM_input_dim, opt.d_feats)
        self.fST_type = vars(opt).get('fST_type', 'fST0')

        self.enc_attn = attention_module_multi_head(opt.d_pos_vec, opt.d_feats, (opt.d_feats, opt.d_feats, opt.d_o), group=opt.n_head, fST_type=self.fST_type)


    def forward(self, feats, soi_select_list):

        num_sois = len(soi_select_list)
        if self.use_posit:
            pos_matrix = self.extract_position_matrix(np.array(soi_select_list), num_sois)
            pos_feats = self.extract_position_embedding(pos_matrix, self.d_pos_vec)
            pos_feats = Variable(torch.FloatTensor(pos_feats), requires_grad=False).cuda()
        else:
            pos_feats = None
        soi_feats = self.event_emb(feats)
        #enc_input = roi_feats + Variable(torch.FloatTensor(pos_feats), requires_grad=False).cuda()
        # [num_rois, fusion_opt.d_o]
        enc_feats = self.enc_attn(soi_feats, pos_feats, self.use_posit)

        return enc_feats

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=10000):
        num_rois, nongt_dim, _ = position_mat.shape
        feat_range = np.arange(0, feat_dim / 4)

        dim_mat = np.power(np.full((1,), wave_length), (4. / feat_dim) * feat_range)
        dim_mat = np.reshape(dim_mat, newshape=(1, 1, 1, -1))
        position_mat = np.expand_dims(100.0 * position_mat, axis=3)
        div_mat = np.divide(position_mat, dim_mat)
        sin_mat = np.sin(div_mat)
        cos_mat = np.cos(div_mat)
        embedding = np.concatenate((sin_mat, cos_mat), axis=3)
        embedding = np.reshape(embedding, newshape=(num_rois, nongt_dim, feat_dim))
        return embedding

    @staticmethod
    def extract_position_matrix(bbox, nongt_dim):
        start, end = np.split(bbox, 2, axis=1)
        center = 0.5 * (start + end)
        length = (end - start).astype('float32')
        delta_center = np.divide(center - np.transpose(center), length)
        delta_center = (np.maximum(np.abs(delta_center), 1e-3))
        delta_length = np.divide(np.transpose(length),length)
        delta_length = np.log(delta_length)
        delta_center = np.expand_dims(delta_center, 2)
        delta_length = np.expand_dims(delta_length, 2)
        position_matrix = np.concatenate((delta_center, delta_length), axis=2)

        return position_matrix


class attention_module_multi_head(nn.Module):
    def __init__(self, pos_emb_dim, roi_emb_dim, dim=(1024, 1024, 1024),
                 group=16,fST_type='fST0'):
        super(attention_module_multi_head,self).__init__()
        self.d_q, self.d_k, self.d_o = dim
        self.dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        self.pos_emb_dim = pos_emb_dim
        self.roi_emb_dim = roi_emb_dim

        self.group = group
        self.fST_type = fST_type
        self.pair_pos_fc1 = nn.Linear(self.pos_emb_dim, self.pos_emb_dim)
        self.pair_pos_fc2 = nn.Linear(self.pos_emb_dim, self.group)
        self.query_1 = nn.Linear(self.roi_emb_dim, self.d_q)
        self.key_1 = nn.Linear(self.roi_emb_dim, self.d_k)
        self.softmax_1 = nn.Softmax(dim=2)
        self.linear_out_1 = nn.Conv2d(in_channels=group * roi_emb_dim, out_channels=self.d_o, kernel_size=(1, 1),
                                      stride=1, groups=group)
        self.dropout = nn.Dropout(0.3)
    def forward(self, roi_feat, position_embedding, use_posit=True):

        num_rois = roi_feat.shape[0]

        if use_posit:
            num_rois, _, emb_dim = position_embedding.shape
            # [num_rois * num_rois, emb_dim]
            position_embedding_reshape = position_embedding.contiguous().view(-1, emb_dim)
            # position_feat_1, [num_rois * num_rois, group]
            position_feat_1 = self.pair_pos_fc1(position_embedding_reshape)

            # # aff_weight, [num_rois, num_rois, group]
            #aff_weight = F.relu(position_feat_1, inplace=True).view(-1, num_rois, self.group)
            aff_weight = self.pair_pos_fc2(F.tanh(position_feat_1)).view(-1, num_rois, self.group)
            # aff_weight, [num_rois, group, num_rois]
            aff_weight = torch.transpose(aff_weight, 1, 2)

        # multi head
        assert self.d_q == self.d_k, 'Matrix multiply requires same dimensions!'

        # [num_rois, d_q]
        q_data = self.query_1(roi_feat)

        # [num_rois, group, d_q/group]
        q_data_batch = q_data.view(-1, self.group, self.d_q / self.group)

        # [group, num_rois, d_q/group]
        q_data_batch = torch.transpose(q_data_batch, 0, 1)

        # [num_rois, d_k]
        k_data = self.key_1(roi_feat)
        # [group, num_rois, d_k/group]
        k_data_batch = k_data.view(-1, self.group, self.d_k / self.group).transpose(1, 0)

        v_data = roi_feat

        # [group, num_rois, num_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        # aff_scale, [group, num_rois, num_rois]
        aff_scale = (1.0 / math.sqrt(float(self.d_k / self.group))) * aff

        # aff_scale, [num_rois, group, num_rois]
        aff_scale = aff_scale.transpose(1, 0)

        if use_posit:
            # weighted_aff, [num_rois,group, num_rois]
            #weighted_aff = torch.log(aff_weight.clamp(min=1e-6)) + aff_scale
            if self.fST_type== 'fST0':
                weighted_aff = aff_weight * aff_scale
            if self.fST_type== 'fST1':
                weighted_aff = aff_weight + aff_scale
            if self.fST_type== 'fST2':
                weighted_aff = torch.log(aff_weight.clamp(min=1e-6)) + aff_scale
            if self.fST_type== 'fST3':
                weighted_aff = aff_weight
        else:
            weighted_aff = aff_scale


        aff_softmax = self.softmax_1(weighted_aff)
        #save_and_plot_realtion_weight(aff_weight, aff_scale, weighted_aff, aff_softmax)
        aff_softmax = self.dropout(aff_softmax)

        #debug_code(aff_weight, aff_scale, aff_softmax)

        # [num_rois * groups, num_rois]
        aff_softmax_reshape = aff_softmax.view(-1, num_rois)
        # output_t, [num_rois * group, roi_emb_dim]
        output_t = aff_softmax_reshape.matmul(v_data)
        # output_t, [num_rois, group * roi_emb_dim, 1, 1]
        output_t = output_t.view(-1, self.group * self.roi_emb_dim, 1, 1)
        # linear_out, [num_rois, self.d_o, 1, 1]
        linear_out = self.linear_out_1(output_t)
        # output, [num_rois, self.d_o]
        output = linear_out.squeeze(3).squeeze(2)

        return output

