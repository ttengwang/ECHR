# This file is based on https://github.com/ruotianluo/ImageCaptioning.pytorch/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

# from .CaptionModel import CaptionModel


class OldModel(nn.Module):
    def __init__(self, opt):
        super(OldModel, self).__init__()

        self.CG_init_feats_type = opt.CG_init_feats_type
        self.opt = opt

        self.vocab_size = opt.CG_vocab_size
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.seq_length = opt.CG_seq_length
        #self.fc_feat_size = opt.fc_feat_size
        #self.att_feat_size = opt.att_feat_size
        self.CG_init_feats_dim = self.decide_init_feats_dim()

        self.ss_prob = 0.0 # Schedule sampling probability
        if self.CG_init_feats_dim:
            self.init_linear = nn.Linear(self.CG_init_feats_dim, self.num_layers * self.rnn_size) # feature to rnn_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        if ('two_stream' in opt.caption_model ):
            logit_input_size = 2*self.rnn_size
        elif ('three_stream_2stream' in opt.caption_model):
            logit_input_size = 2*self.rnn_size
        elif ('three_stream' in opt.caption_model ):
            logit_input_size = 3*self.rnn_size
        elif ('H3_dense' in opt.caption_model ):
            logit_input_size = 3*self.rnn_size
        else:
            logit_input_size = self.rnn_size
        self.logit = nn.Linear(logit_input_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def decide_init_feats_dim(self):
        dim = 0
        if 'V' in self.CG_init_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_init_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_init_feats_type:
            dim += self.opt.clip_context_dim
        return dim

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, video, event, clip):
        batch_size = event.shape[0] if event is not None else clip.shape[0]

        if self.CG_init_feats_dim == 0:
            weight = next(self.parameters()).data
            return (Variable(weight.new(self.num_layers, batch_size, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, batch_size, self.rnn_size).zero_()))
        else:
            input_feats = []

            if 'V' in self.CG_init_feats_type:

                video = video.unsqueeze(0).expand([batch_size, self.opt.video_context_dim])
                input_feats.append(video)
            if 'E' in self.CG_init_feats_type:
                input_feats.append(event)
            if 'C' in self.CG_init_feats_type:
                input_feats.append(clip.mean(1))

            input_feats = torch.cat(input_feats,1)
            image_map = self.init_linear(input_feats).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
            if self.rnn_type == 'lstm':
                return (image_map, image_map)
            else:
                return image_map

    def forward(self, video, event , clip, clip_mask, seq):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        state = self.init_hidden(video, event, clip)
        outputs = []

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = puppet.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.get_logprobs_state(it, video, event , clip, clip_mask, state)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)


    def get_logprobs_state(self, it, video, event , clip, clip_mask, state):
        xt = self.embed(it)
        output, state = self.core(xt, video, event , clip, clip_mask, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)))
        return logprobs, state

    def sample(self, video, event , clip, clip_mask, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            assert  AssertionError, 'beamsize > 1'
            #return self.sample_beam(fc_feats, att_feats, opt)

        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]
        state = self.init_hidden(video, event, clip)

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = puppet.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cuda() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cuda()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing


            logprobs, state = self.get_logprobs_state(it, video, event , clip, clip_mask, state)

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

        if seq==[] or len(seq)==0:
            return [],[]
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


class ShowAttendTellCore(nn.Module):

    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        #self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        self.opt = opt
        self.CG_input_feats_type = opt.CG_input_feats_type
        self.CG_input_dim = self.decide_input_feats_dim()

        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.CG_input_dim,
                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1)
        else:
            self.ctx2att = nn.Linear(self.att_feat_size, 1)
            self.h2att = nn.Linear(self.rnn_size, 1)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim

    def get_input_feats(self, video, event, att_clip):
        puppet = event if (event is not None) else att_clip
        batch_size = puppet.shape[0]
        input_feats = []

        if 'V' in self.CG_input_feats_type:
            video = video.unsqueeze(0).expand([batch_size, self.opt.video_context_dim])
            input_feats.append(video)
        if 'E' in self.CG_input_feats_type:
            input_feats.append(event)
        if 'C' in self.CG_input_feats_type:
            input_feats.append(att_clip)

        input_feats = torch.cat(input_feats,1)
        return input_feats

    def forward(self, xt, video, event, clip, clip_mask, state):
        att_size = clip.numel() // clip.size(0) // self.opt.clip_context_dim
        att = clip.view(-1, self.opt.clip_context_dim)
        if self.att_hid_size > 0:
            att = self.ctx2att(att)                             # (batch * att_size) * att_hid_size
            att = att.view(-1, att_size, self.att_hid_size)     # batch * att_size * att_hid_size
            att_h = self.h2att(state[0][-1])                    # batch * att_hid_size
            att_h = att_h.unsqueeze(1).expand_as(att)           # batch * att_size * att_hid_size
            dot = att + att_h                                   # batch * att_size * att_hid_size
            dot = F.tanh(dot)                                   # batch * att_size * att_hid_size
            dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
            dot = self.alpha_net(dot)                           # (batch * att_size) * 1
            dot = dot.view(-1, att_size)                        # batch * att_size
        else:
            att = self.ctx2att(att)(att)                        # (batch * att_size) * 1
            att = att.view(-1, att_size)                        # batch * att_size
            att_h = self.h2att(state[0][-1])                    # batch * 1
            att_h = att_h.expand_as(att)                        # batch * att_size
            dot = att_h + att                                   # batch * att_size

        weight = F.softmax(dot)
        if clip_mask is not None:
            weight = weight * clip_mask.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)

        att_feats_ = clip.view(-1, att_size, self.att_feat_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        input_feats = self.get_input_feats(video, event, att_res)
        output, state = self.rnn(torch.cat([xt, input_feats], 1).unsqueeze(0), state)
        return output.squeeze(0), state

class AllImgCore(nn.Module):
    def __init__(self, opt):
        super(AllImgCore, self).__init__()
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size

        self.opt = opt
        self.CG_input_feats_type = opt.CG_input_feats_type
        self.CG_input_dim = self.decide_input_feats_dim()

        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.CG_input_dim,
                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim

    def get_input_feats(self, video, event, clip):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]
        input_feats = []

        if 'V' in self.CG_input_feats_type:
            video = video.unsqueeze(0).expand([batch_size, self.opt.video_context_dim])
            input_feats.append(video)
        if 'E' in self.CG_input_feats_type:
            input_feats.append(event)
        if 'C' in self.CG_input_feats_type:
            input_feats.append(clip.mean(1))

        input_feats = torch.cat(input_feats,1)
        return input_feats


    def forward(self, xt, video, event, clip, clip_mask, state):
        input_feats = self.get_input_feats(video, event, clip)
        output, state = self.rnn(torch.cat([xt, input_feats], 1).unsqueeze(0), state)
        return output.squeeze(0), state


class H3_basic(nn.Module):
    def __init__(self, opt):
        super(H3_basic, self).__init__()
        self.opt =opt
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        #self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        self.CG_input_feats_type = opt.CG_input_feats_type
        self.CG_input_dim = self.decide_input_feats_dim()
        #        self.rnn = (self.input_encoding_size + self.fc_feat_size -200+ self.att_feat_size,
        #                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1)

        else:
            self.ctx2att = nn.Linear(self.att_feat_size, 1)
            self.h2att = nn.Linear(self.rnn_size, 1)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim




class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.CG_rnn_size
        self.att_hid_size = opt.CG_att_hid_size
        self.att_feat_size = opt.clip_context_dim
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, att_mask):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = att_feats.view(-1, self.att_feat_size)
        #pdb.set_trace()
        att = self.ctx2att(att)
        att = att.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1

        #pdb.set_trace()
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot)                             # batch * att_size
        if att_mask is not None:
            weight = weight * att_mask.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)

        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
        return att_res


class H3_Core(H3_basic):
    def __init__(self, opt):
        super(H3_Core,self).__init__(opt)
        self.layer0 = nn.LSTMCell(self.opt.video_context_dim + self.rnn_size + self.input_encoding_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.event_context_dim + self.rnn_size, self.rnn_size)
        self.layer2 = nn.LSTMCell(self.opt.clip_context_dim + self.rnn_size, self.rnn_size)
        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)

    def forward(self, xt, video, event ,clip ,clip_mask, state):
            puppet = event if (event is not None) else clip
            batch_size = puppet.shape[0]

            pre_h = state[0][-1]
            #pdb.set_trace()
            video = video.unsqueeze(0).expand([batch_size,self.opt.video_context_dim])
            layer0_input = torch.cat([xt,video,pre_h],1)
            h0,c0 = self.layer0(layer0_input, (state[0][0], state[1][0]))
            h0 = self.dropout0(h0)

            layer1_input = torch.cat([event, h0], 1)
            h1,c1 = self.layer1(layer1_input, (state[0][1], state[1][1]))
            h1 = self.dropout1(h1)

            att = self.attention(h1, clip, clip_mask)
            layer2_input = torch.cat([att,h1],1)
            h2,c2= self.layer2(layer2_input, (state[0][2],state[1][2]))

            #output = self.dropout1(h2)
            state = (torch.stack((h0,h1,h2)), torch.stack((c0,c1,c2)))

            return h2, state

class H3_dense_Core(H3_basic):
    def __init__(self, opt):
        super(H3_dense_Core,self).__init__(opt)
        self.layer0 = nn.LSTMCell(self.opt.video_context_dim + self.rnn_size + self.input_encoding_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.event_context_dim + self.rnn_size, self.rnn_size)
        self.layer2 = nn.LSTMCell(self.opt.clip_context_dim + self.rnn_size, self.rnn_size)
        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)

    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        pre_h = state[0][-1]
        #pdb.set_trace()
        video = video.unsqueeze(0).expand([batch_size,self.opt.video_context_dim])
        layer0_input = torch.cat([xt,video,pre_h],1)
        _h0,c0 = self.layer0(layer0_input, (state[0][0], state[1][0]))
        h0 = self.dropout0(_h0)

        layer1_input = torch.cat([event, h0], 1)
        _h1,c1 = self.layer1(layer1_input, (state[0][1], state[1][1]))
        h1 = self.dropout1(_h1)

        att = self.attention(h1, clip, clip_mask)
        layer2_input = torch.cat([att,h1],1)
        h2,c2= self.layer2(layer2_input, (state[0][2],state[1][2]))

        #output = self.dropout1(h2)
        state = (torch.stack((h0,h1,h2)), torch.stack((c0,c1,c2)))

        output = torch.cat((_h0,_h1,h2),1)
        return output, state


class H3_dense_add_Core(H3_basic):
    def __init__(self, opt):
        super(H3_dense_add_Core,self).__init__(opt)
        self.layer0 = nn.LSTMCell(self.opt.video_context_dim + self.rnn_size + self.input_encoding_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.event_context_dim + self.rnn_size, self.rnn_size)
        self.layer2 = nn.LSTMCell(self.opt.clip_context_dim + self.rnn_size, self.rnn_size)
        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)

    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        pre_h = state[0][-1]
        #pdb.set_trace()
        video = video.unsqueeze(0).expand([batch_size,self.opt.video_context_dim])
        layer0_input = torch.cat([xt,video,pre_h],1)
        _h0,c0 = self.layer0(layer0_input, (state[0][0], state[1][0]))
        h0 = self.dropout0(_h0)

        layer1_input = torch.cat([event, h0], 1)
        _h1,c1 = self.layer1(layer1_input, (state[0][1], state[1][1]))
        h1=_h1+h0
        h1 = self.dropout1(h1)

        att = self.attention(h1, clip, clip_mask)
        layer2_input = torch.cat([att,h1],1)
        _h2,c2= self.layer2(layer2_input, (state[0][2],state[1][2]))
        h2=_h2+h1
        #output = self.dropout1(h2)
        state = (torch.stack((_h0,_h1,_h2)), torch.stack((c0,c1,c2)))


        return h2, state


class TwoStream_Core(nn.Module):
    def __init__(self, opt):
        super(TwoStream_Core, self).__init__()
        self.opt =opt
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        #self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        self.CG_input_feats_type = opt.CG_input_feats_type
        self.CG_input_dim = self.decide_input_feats_dim()
        #        self.rnn = (self.input_encoding_size + self.fc_feat_size -200+ self.att_feat_size,
        #                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        self.layer0 = nn.LSTMCell(self.opt.event_context_dim + self.input_encoding_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.clip_context_dim + self.input_encoding_size, self.rnn_size)
        self.fusion_layer = nn.Linear(self.rnn_size*2, self.rnn_size)


        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)
        self.dropout_fusion = nn.Dropout(0.5)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim


    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        pre_h0 = state[0][0]
        pre_h1 = state[0][1]

        stream0_input = torch.cat((xt, event),1)
        h0,c0 = self.layer0(stream0_input, (state[0][0], state[1][0]))
        h0 = self.dropout0(h0)
        att = self.attention(pre_h1, clip, clip_mask)
        stream1_input = torch.cat((xt, att),1)
        h1,c1 = self.layer1(stream1_input, (state[0][1], state[1][1]))
        h1=self.dropout1(h1)
        #output = self.dropout1(h2)
        state = (torch.stack((h0,h1)), torch.stack((c0,c1)))
        output = torch.cat((h0,h1),1)
        return output, state

class ThreeStream_Core_2stream(nn.Module):
    def __init__(self, opt):
        super(ThreeStream_Core_2stream, self).__init__()
        self.opt =opt
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        #self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        #self.CG_input_feats_type = opt.CG_input_feats_type
        # self.CG_input_dim = self.decide_input_feats_dim()
        #        self.rnn = (self.input_encoding_size + self.fc_feat_size -200+ self.att_feat_size,
        #                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        self.layer0 = nn.LSTMCell(self.opt.event_context_dim + self.input_encoding_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.clip_context_dim + self.input_encoding_size, self.rnn_size)
        #self.layer2 = nn.LSTMCell(self.opt.video_context_dim + self.input_encoding_size, self.rnn_size)
        #self.fusion_layer = nn.Linear(self.rnn_size*2, self.rnn_size)

        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim


    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        pre_h1 = state[0][1]
        video = video.unsqueeze(0).expand([batch_size,self.opt.video_context_dim])

        stream0_input = torch.cat((xt, event),1)
        h0,c0 = self.layer0(stream0_input, (state[0][0], state[1][0]))
        h0 = self.dropout0(h0)
        att = self.attention(pre_h1, clip, clip_mask)
        stream1_input = torch.cat((xt, att),1)
        h1,c1 = self.layer1(stream1_input, (state[0][1], state[1][1]))
        h1=self.dropout1(h1)

        #stream2_input = torch.cat((xt, video),1)
        #h2,c2 = self.layer2(stream2_input, (state[0][2], state[1][2]))
        #h2=self.dropout2(h2)

        #output = self.dropout1(h2)
        state = (torch.stack((h0,h1)), torch.stack((c0,c1)))
        output = torch.cat((h0,h1),1)
        return output, state


class ThreeStream_Core_2stream_CC(nn.Module):
    def __init__(self, opt):
        super(ThreeStream_Core_2stream_CC, self).__init__()
        self.opt =opt
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        #self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        #self.CG_input_feats_type = opt.CG_input_feats_type
        # self.CG_input_dim = self.decide_input_feats_dim()
        #        self.rnn = (self.input_encoding_size + self.fc_feat_size -200+ self.att_feat_size,
        #                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        self.layer0 = nn.LSTMCell(self.opt.clip_context_dim + self.input_encoding_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.clip_context_dim + self.input_encoding_size, self.rnn_size)
        #self.layer2 = nn.LSTMCell(self.opt.video_context_dim + self.input_encoding_size, self.rnn_size)
        #self.fusion_layer = nn.Linear(self.rnn_size*2, self.rnn_size)

        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim


    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        pre_h1 = state[0][1]
        att = self.attention(pre_h1, clip, clip_mask)

        # video = video.unsqueeze(0).expand([batch_size,self.opt.video_context_dim])

        stream0_input = torch.cat((xt, att),1)
        h0,c0 = self.layer0(stream0_input, (state[0][0], state[1][0]))
        h0 = self.dropout0(h0)

        stream1_input = torch.cat((xt, att),1)
        h1,c1 = self.layer1(stream1_input, (state[0][1], state[1][1]))
        h1=self.dropout1(h1)

        #stream2_input = torch.cat((xt, video),1)
        #h2,c2 = self.layer2(stream2_input, (state[0][2], state[1][2]))
        #h2=self.dropout2(h2)

        #output = self.dropout1(h2)
        state = (torch.stack((h0,h1)), torch.stack((c0,c1)))
        output = torch.cat((h0,h1),1)
        return output, state


class ThreeStream_Core_2stream_CLDA(nn.Module):
    def __init__(self, opt):
        super(ThreeStream_Core_2stream_CLDA, self).__init__()
        self.opt =opt
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        #self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        #self.CG_input_feats_type = opt.CG_input_feats_type
        # self.CG_input_dim = self.decide_input_feats_dim()
        #        self.rnn = (self.input_encoding_size + self.fc_feat_size -200+ self.att_feat_size,
        #                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        self.layer0 = nn.LSTMCell(self.opt.video_context_dim + self.input_encoding_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.clip_context_dim + self.input_encoding_size, self.rnn_size)
        #self.layer2 = nn.LSTMCell(self.opt.video_context_dim + self.input_encoding_size, self.rnn_size)
        #self.fusion_layer = nn.Linear(self.rnn_size*2, self.rnn_size)

        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim


    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        pre_h1 = state[0][1]
        video = video.unsqueeze(0).expand([batch_size,self.opt.video_context_dim])

        stream0_input = torch.cat((xt, video),1)
        h0,c0 = self.layer0(stream0_input, (state[0][0], state[1][0]))
        h0 = self.dropout0(h0)
        att = self.attention(pre_h1, clip, clip_mask)
        stream1_input = torch.cat((xt, att),1)
        h1,c1 = self.layer1(stream1_input, (state[0][1], state[1][1]))
        h1=self.dropout1(h1)

        #stream2_input = torch.cat((xt, video),1)
        #h2,c2 = self.layer2(stream2_input, (state[0][2], state[1][2]))
        #h2=self.dropout2(h2)

        #output = self.dropout1(h2)
        state = (torch.stack((h0,h1)), torch.stack((c0,c1)))
        output = torch.cat((h0,h1),1)
        return output, state

class ThreeStream_Core(nn.Module):
    def __init__(self, opt):
        super(ThreeStream_Core, self).__init__()
        self.opt =opt
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        #self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        self.CG_input_feats_type = opt.CG_input_feats_type
        self.CG_input_dim = self.decide_input_feats_dim()
        #        self.rnn = (self.input_encoding_size + self.fc_feat_size -200+ self.att_feat_size,
        #                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        self.layer0 = nn.LSTMCell(self.opt.event_context_dim + self.input_encoding_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.clip_context_dim + self.input_encoding_size, self.rnn_size)
        self.layer2 = nn.LSTMCell(self.opt.video_context_dim + self.input_encoding_size, self.rnn_size)
        self.fusion_layer = nn.Linear(self.rnn_size*3, self.rnn_size)

        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim


    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        pre_h1 = state[0][1]
        video = video.unsqueeze(0).expand([batch_size,self.opt.video_context_dim])

        stream0_input = torch.cat((xt, event),1)
        h0,c0 = self.layer0(stream0_input, (state[0][0], state[1][0]))
        h0 = self.dropout0(h0)
        att = self.attention(pre_h1, clip, clip_mask)
        stream1_input = torch.cat((xt, att),1)
        h1,c1 = self.layer1(stream1_input, (state[0][1], state[1][1]))
        h1=self.dropout1(h1)

        stream2_input = torch.cat((xt, video),1)
        h2,c2 = self.layer2(stream2_input, (state[0][2], state[1][2]))
        h2=self.dropout2(h2)

        #output = self.dropout1(h2)
        state = (torch.stack((h0,h1,h2)), torch.stack((c0,c1,c2)))
        output = torch.cat((h0,h1,h2),1)
        return output, state


class LMResBlock(nn.Module):
    def __init__(self, opt):
        super(ThreeStream_Core, self).__init__()
        self.opt =opt
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        #self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        self.CG_input_feats_type = opt.CG_input_feats_type
        self.CG_input_dim = self.decide_input_feats_dim()
        #        self.rnn = (self.input_encoding_size + self.fc_feat_size -200+ self.att_feat_size,
        #                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        self.layer0 = nn.LSTMCell(self.opt.event_context_dim + self.input_encoding_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.clip_context_dim + self.input_encoding_size, self.rnn_size)

        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim


    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event
        batch_size = puppet.shape[0]
        pre_h1 = state[0][1]
        video = video.unsqueeze(0).expand([batch_size,self.opt.video_context_dim])

        stream0_input = torch.cat((xt, event),1)
        h0,c0 = self.layer0(stream0_input, (state[0][0], state[1][0]))

        h0 = self.dropout0(h0)
        att = self.attention(pre_h1, clip, clip_mask)
        stream1_input = torch.cat((xt, att),1)
        h1,c1 = self.layer1(stream1_input, (state[0][1], state[1][1]))
        h1=self.dropout1(h1)

        stream2_input = torch.cat((xt, video),1)
        h2,c2 = self.layer2(stream2_input, (state[0][2], state[1][2]))
        h2=self.dropout2(h2)

        #output = self.dropout1(h2)
        state = (torch.stack((h0,h1,h2)), torch.stack((c0,c1,c2)))
        output = torch.cat((h0,h1,h2),1)
        return output, state


class TwoStream_jump_Core(nn.Module):
    def __init__(self, opt):
        super(TwoStream_jump_Core, self).__init__()
        self.opt =opt
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        #self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        self.CG_input_feats_type = opt.CG_input_feats_type
        self.CG_input_dim = self.decide_input_feats_dim()

        self.layer0 = nn.LSTMCell(self.opt.event_context_dim + self.input_encoding_size + self.rnn_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.clip_context_dim + self.input_encoding_size + self.rnn_size, self.rnn_size)
        self.fusion_layer = nn.Linear(self.rnn_size*2, self.rnn_size)


        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)
        self.dropout_fusion = nn.Dropout(0.5)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim


    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        pre_h0 = state[0][0]
        pre_h1 = state[0][1]

        stream0_input = torch.cat((xt, event, pre_h1),1)
        h0,c0 = self.layer0(stream0_input, (state[0][0], state[1][0]))
        h0 = self.dropout0(h0)
        att = self.attention(pre_h1, clip, clip_mask)
        stream1_input = torch.cat((xt, att, pre_h0),1)
        h1,c1 = self.layer1(stream1_input, (state[0][1], state[1][1]))
        h1 = self.dropout1(h1)
        state = (torch.stack((h0,h1)), torch.stack((c0,c1)))
        output = torch.cat((h0,h1),1)
        return output, state


class TwoStream3LSTM_Core(nn.Module):
    def __init__(self, opt):
        super(TwoStream3LSTM_Core, self).__init__()
        self.opt =opt
        self.input_encoding_size = opt.CG_input_encoding_size
        self.rnn_type = opt.CG_rnn_type
        self.rnn_size = opt.CG_rnn_size
        #self.num_layers = opt.CG_num_layers
        self.drop_prob_lm = opt.CG_drop_prob
        self.fc_feat_size = opt.CG_fc_feat_size
        self.att_feat_size = opt.clip_context_dim
        self.att_hid_size = opt.CG_att_hid_size

        self.CG_input_feats_type = opt.CG_input_feats_type
        self.CG_input_dim = self.decide_input_feats_dim()
        #        self.rnn = (self.input_encoding_size + self.fc_feat_size -200+ self.att_feat_size,
        #                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        self.layer0 = nn.LSTMCell(self.opt.event_context_dim + self.rnn_size, self.rnn_size)
        self.layer1 = nn.LSTMCell(self.opt.clip_context_dim + self.rnn_size, self.rnn_size)
        self.layer2 = nn.LSTMCell(self.opt.video_context_dim + self.input_encoding_size, self.rnn_size)
        self.fusion_layer = nn.Linear(self.rnn_size*2, self.rnn_size)


        self.attention = Attention(opt)
        self.dropout0=nn.Dropout(0.5)
        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)
        self.dropout_fusion = nn.Dropout(0.5)

    def decide_input_feats_dim(self):
        dim = 0
        if 'V' in self.CG_input_feats_type:
            dim += self.opt.video_context_dim
        if 'E' in self.CG_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.CG_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim


    def forward(self, xt, video, event ,clip ,clip_mask, state):
        puppet = event if (event is not None) else clip
        batch_size = puppet.shape[0]

        pre_h1 = state[0][1]
        video = video.unsqueeze(0).expand([batch_size,self.opt.video_context_dim])

        base_stream_input = torch.cat((xt, video), 1)
        h2,c2 = self.layer2(base_stream_input, (state[0][2], state[1][2]))
        h2 = self.dropout2(h2)

        stream0_input = torch.cat((h2, event),1)
        h0,c0 = self.layer0(stream0_input, (state[0][0], state[1][0]))
        h0 = self.dropout0(h0)
        att = self.attention(pre_h1, clip, clip_mask)
        stream1_input = torch.cat((h2, att),1)
        h1,c1 = self.layer1(stream1_input, (state[0][1], state[1][1]))
        h1=self.dropout1(h1)
        #output = self.dropout1(h2)
        state = (torch.stack((h0,h1,h2)), torch.stack((c0,c1,c2)))
        output = torch.cat((h0,h1),1)
        return output, state


class ShowAttendTellModel(OldModel):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)

class AllImgModel(OldModel):
    def __init__(self, opt):
        super(AllImgModel, self).__init__(opt)
        self.core = AllImgCore(opt)

class H3Model(OldModel):
    def __init__(self, opt):
        super(H3Model, self).__init__(opt)
        self.core = H3_Core(opt)

class H3denseModel(OldModel):
    def __init__(self, opt):
        super(H3denseModel, self).__init__(opt)
        self.core = H3_dense_Core(opt)

class H3denaddModel(OldModel):
    def __init__(self, opt):
        super(H3denaddModel, self).__init__(opt)
        self.core = H3_dense_add_Core(opt)


class TwostreamModel(OldModel):
    def __init__(self, opt):
        super(TwostreamModel, self).__init__(opt)
        self.core = TwoStream_Core(opt)

class ThreestreamModel(OldModel):
    def __init__(self, opt):
        super(ThreestreamModel, self).__init__(opt)
        self.core = ThreeStream_Core(opt)

class ThreestreamModel_2stream(OldModel):
    def __init__(self, opt):
        super(ThreestreamModel_2stream, self).__init__(opt)
        self.core = ThreeStream_Core_2stream(opt)

class ThreestreamModel_2stream_LDA(OldModel):
    def __init__(self, opt):
        super(ThreestreamModel_2stream_LDA, self).__init__(opt)
        self.core = ThreeStream_Core_2stream_CLDA(opt)

class ThreestreamModel_2stream_CC(OldModel):
    def __init__(self, opt):
        super(ThreestreamModel_2stream_CC, self).__init__(opt)
        self.core = ThreeStream_Core_2stream_CC(opt)

class TwostreamModel_3LSTM(OldModel):
    def __init__(self, opt):
        super(TwostreamModel_3LSTM, self).__init__(opt)
        self.core = TwoStream3LSTM_Core(opt)


class Twostream_jump_Model(OldModel):
    def __init__(self, opt):
        super(Twostream_jump_Model, self).__init__(opt)
        self.core = TwoStream_jump_Core(opt)
