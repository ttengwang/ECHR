import pdb
import torch
from torch import nn
from torch.autograd import Variable
import models
import time
class CaptionGenerator(nn.Module):
    def __init__(self, opt):
        super(CaptionGenerator, self).__init__()
        self.opt = opt
        self.change_context_dim()
        if 'TSRM' in opt.fusion_model and 'ER' in opt.event_context_type:
            self.fusion_model = models.setup_fusion(opt)
        self.lm_model = models.setup_lm(opt)


    def forward(self, tap_feats, c3d_feats, lda_feats, lm_labels, ind_select_list, soi_select_list, mode='train'):
        '''
        VL+VC+VH, EC+EH+ER1+ER2+ER3, CL+CC+CH
        '''

        t0=time.time()
        video = self.get_video_context( tap_feats, c3d_feats, lda_feats, ind_select_list, soi_select_list)
        event = self.get_event_context( tap_feats, c3d_feats, lda_feats, ind_select_list, soi_select_list)
        clip, clip_mask = self.get_clip_context( tap_feats, c3d_feats, lda_feats, ind_select_list, soi_select_list)


        t1=time.time()
        if mode=='train':
            pred_captions = self.lm_model(video, event, clip, clip_mask, lm_labels)
            return pred_captions
        elif mode =='train_rl':
            gen_result, sample_logprobs = self.lm_model.sample(video, event,clip, clip_mask,opt={'sample_max': 0})
            self.lm_model.eval()
            with torch.no_grad():
                greedy_res, _ = self.lm_model.sample(video, event,clip, clip_mask)
            self.lm_model.train()
            return gen_result, sample_logprobs, greedy_res
        elif mode == 'eval':
            seq, cg_prob = self.lm_model.sample(video, event, clip, clip_mask)

            t2= time.time()
            print(t1 - t0, t2 - t1)
            return seq, cg_prob
        elif mode == '1stage':
            pred_captions, cg_feats = self.lm_model(video, event,clip, clip_mask, lm_labels, need_ext_data=False)
            return pred_captions, cg_feats
        elif mode == '1stage_eval':
            seq, cg_prob, cg_feats = self.lm_model.sample(video, event, clip, clip_mask, need_ext_data=False)
            return seq, cg_prob, cg_feats
        elif mode == '1stage_ATTnorm':
            pred_captions, cg_feats, (att_weights, att_mask) = self.lm_model(video, event, clip, clip_mask, lm_labels, need_ext_data=True)
            return pred_captions, cg_feats, (att_weights, att_mask)


    def change_context_dim(self):
        opt = self.opt
        video_context_dim = 0
        if 'VL' in opt.video_context_type:
            video_context_dim += opt.lda_dim
        if 'VC' in opt.video_context_type:
            video_context_dim += opt.video_dim
        if 'VH' in opt.video_context_type:
            video_context_dim += opt.hidden_dim

        event_context_dim = 0

        if 'ER' in opt.event_context_type:
            event_context_dim = opt.d_o
        else:
            if 'EC' in opt.event_context_type:
                event_context_dim += opt.video_dim
            if 'EH' in opt.event_context_type:
                event_context_dim += opt.hidden_dim

        clip_context_dim = 0
        if 'CC' in opt.clip_context_type:
            clip_context_dim += opt.video_dim
        if 'CH' in opt.clip_context_type:
            clip_context_dim += opt.hidden_dim

        opt.video_context_dim = video_context_dim
        opt.event_context_dim = event_context_dim
        opt.clip_context_dim = clip_context_dim


    def get_video_context(self, tap_feats, c3d_feats, lda_feats, ind_select_list, soi_select_list):
        opt = self.opt
        video_feats = []
        if 'VL' in opt.video_context_type:
            video_feats.append(lda_feats)

        if 'VC' in opt.video_context_type:
            video_feats.append(c3d_feats.mean(0))

        if 'VH' in opt.video_context_type:
            video_feats.append(tap_feats.mean(0))

        if video_feats:
            video_feat = torch.cat(video_feats, 0)
        else:
            video_feat = None

        return video_feat

    def get_event_context(self, tap_feats, c3d_feats, lda_feats, ind_select_list, soi_select_list):
        opt=self.opt
        event_feats =[]
        if 'EC' in opt.event_context_type or 'ER1' in opt.event_context_type or 'ER3' in opt.event_context_type:
            pooled = []
            for i, soi in enumerate(soi_select_list):
                selected = c3d_feats[soi[0]:soi[1]]
                pooled.append(selected.mean(0).unsqueeze(0))
            EC = torch.cat(pooled, 0)
            event_feats.append(EC)
            if 'ER1' in opt.event_context_type:
                event_feat = self.fusion_model(EC, soi_select_list)
                return event_feat

        if 'EH' in opt.event_context_type or 'ER2' in opt.event_context_type or 'ER3' in opt.event_context_type:
            EH = tap_feats[ind_select_list]
            event_feats.append(EH)
            if 'ER2' in opt.event_context_type:
                event_feat = self.fusion_model(EH, soi_select_list)
                return event_feat

        if 'ER3' in opt.event_context_type:
            ECH = torch.cat((EC,EH),1)
            event_feat = self.fusion_model(ECH, soi_select_list)
            return  event_feat

        if event_feats:
            event_feat = torch.cat(event_feats, 0)
        else:
            event_feat = None

        return event_feat


    def get_clip_context(self, tap_feats, c3d_feats, lda_feats, ind_select_list, soi_select_list):
        opt = self.opt
        max_att_len = max([(s[1]-s[0]) for s in soi_select_list])
        clip_mask = Variable(c3d_feats.new(len(soi_select_list), max_att_len).zero_())
        clip_feats = []

        if 'CC' in opt.clip_context_type:
            CC = Variable(c3d_feats.new(len(soi_select_list), max_att_len, opt.video_dim).zero_())
            for i,soi in enumerate(soi_select_list):
                selected = c3d_feats[soi[0]:soi[1]]
                CC[i,:len(selected),:] = selected
                clip_mask[i, :len(selected)] = 1
            clip_feats.append(CC)

        if 'CH' in opt.clip_context_type:
            CH = Variable(c3d_feats.new(len(soi_select_list), max_att_len, opt.hidden_dim).zero_())
            for i,soi in enumerate(soi_select_list):
                selected = tap_feats[soi[0]:soi[1]]
                CH[i,:len(selected),:] = selected
                clip_mask[i, :len(selected)] = 1
            clip_feats.append(CH)

        if clip_feats:
            clip_feat = torch.cat(clip_feats, 2)
        else:
            clip_feat = None
            clip_mask = None
        return clip_feat, clip_mask

