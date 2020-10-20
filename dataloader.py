# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import cPickle
import h5py
import os, time, pdb
import numpy as np
import random
import torch
import torch.utils.data as data

import multiprocessing
import pandas as pd

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split,
                                                    self, (split == 'train') and (self.opt.shuffle))
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_dataset_dize(self, mode):
        return len(self.split_ix[mode])

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_other_feats(self, other_features):
        other_feats = {'lda': None}
        if 'lda' in other_features:
            lda_file = h5py.File(self.opt.input_lda_path, 'r')
            lda_data = {vid: lda_file[vid].value for vid in lda_file.keys()}
            lda_file.close()
            other_feats['lda'] = lda_data
        return other_feats

    def get_c3d_feature(self, video_id):
        feature = np.load(os.path.join(self.input_c3d_dir2, video_id + '.npy')).astype('float32')
        mean = -0.001915027447565527
        var = 1.9239444588254049
        feature = (feature - mean) / np.sqrt(var)
        att_feature = np.zeros((1, 1, 1)).astype('float32')
        return feature, att_feature

    def get_twostream_feature(self, video_id):
        path = os.path.join(self.opt.input_twostream_dir, 'spatial', 'csv_action', video_id + '.csv')
        if not os.path.exists(path):
            vid_len = np.load(os.path.join(self.input_c3d_dir2, video_id + '.npy')).astype('float32').shape[0]
            att_feature = np.zeros((1, 1, 1)).astype('float32')
            return np.zeros((vid_len, 400)), att_feature
        spatial = pd.read_csv(path)
        OF = pd.read_csv(os.path.join(self.opt.input_twostream_dir, 'OF', 'csv_action', video_id + '.csv'))
        if spatial.shape[0] >= OF.shape[0]:
            vid_len = OF.shape[0]
        else:
            vid_len = spatial.shape[0]
        feature = np.concatenate((spatial[:vid_len], OF[:vid_len]),1)
        att_feature = np.zeros((1, 1, 1)).astype('float32')
        return feature,att_feature


    def get_data(self, ix):

        video_id = self.info['videos'][ix]['video_id']

        # feature = np.array(self.feats_c3d[video_id]['c3d_features']).astype('float32')

        features, att_features = [], []
        if vars(self.opt).get('use_c3d_feature',True):
            feature1, att_feature1 = self.get_c3d_feature(video_id)
            features.append(feature1)
            att_features.append(att_feature1)

        if vars(self.opt).get('use_2stream_feature',False):
            feature2, att_feature2 = self.get_twostream_feature(video_id)
            feature2 = feature2[::2]
            att_feature2 = att_feature2[::2]
            features.append(feature2)
            att_features.append(att_feature2)

        vid_len = 1e10
        for f in features:
            vid_len = f.shape[0] if f.shape[0] < vid_len else vid_len
        features = [f[:vid_len] for f in features]
        feature = np.concatenate(features, 1).astype('float32')
        att_feature = np.concatenate(att_features, 1).astype('float32')


        iou_scores, tap_masks, gts_index, gt_featstamps, tap_other = self.get_vid_data(video_id, feature.shape[0])
        if self.use_SOTA_tep:
            SOTA_featstamps, SOTA_Prop_score, SOTA_timestamps = self.get_SOTA_TEP_label(video_id, feature.shape[0])
        else:
            SOTA_featstamps = SOTA_Prop_score = SOTA_timestamps = None

        w1 = np.array(self.w1).astype('float32')

        tap_labels = (iou_scores >= self.opt.iou_threshold)

        tap_masks_good_proposal = (iou_scores >= self.opt.iou_threshold_for_good_proposal)  # * tap_masks

        lda_feat = np.array(self.other_feats['lda'][video_id]).astype('float32') if self.opt.use_lda else np.array(
            [0])

        other = {}
        train_only = {}

        other['gt_featstamps'] = gt_featstamps
        other['SOTA_featstamps'] = SOTA_featstamps
        other['SOTA_timestamps'] = SOTA_timestamps
        other['SOTA_Prop_score'] = SOTA_Prop_score

        # if ix < self.train_length:  # if ix is in training set
        if True:
            tap_gts_for_good_proposal = (tap_masks_good_proposal * (gts_index + 1) - 1).astype('int')
            proposal_num = (tap_gts_for_good_proposal >= 0).sum()
            # assert ncap == tap_gts_for_good_proposal.max() + 1
            other['tap_gts_for_good_proposal'] = tap_gts_for_good_proposal

            if self.opt.tap_model == "sst_1stage" and proposal_num > 0:
                tap_list, lm_list, soi_list, sampled_ids, action_label = self.get_shuffle_list(tap_gts_for_good_proposal,gt_featstamps,
                                                                                 method='1stage')
                other['action_label'] = action_label
            else:
                tap_list, lm_list, soi_list, sampled_ids = self.get_shuffle_list(tap_gts_for_good_proposal,gt_featstamps,
                                                                                 method='random')

            train_only['ind_select_list'] = np.array(tap_list[sampled_ids]).astype('int')  # sampled
            train_only['ind_select_list_eval'] = np.array(tap_list).astype('int')  # sampled
            train_only['cg_select_list'] = np.array(lm_list[sampled_ids]).astype('int')  # sampled
            train_only['soi_select_list'] = np.array(soi_list[sampled_ids]).astype('int')  # sampled
            train_only['soi_select_list_eval'] = np.array(soi_list).astype('int')  # sampled
            train_only['sampled_ids'] = np.array(sampled_ids).astype('int')


        return (feature,
                lda_feat,
                att_feature,
                tap_labels,
                tap_masks,
                iou_scores,
                gts_index,
                tap_masks_good_proposal,
                train_only,
                # tap_good_proposal_info,
                w1,
                ix,
                other)

    def __init__(self, opt):
        # initial some variables
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.use_att = getattr(opt, 'use_att', False)
        self.iou_threshold = self.opt.iou_threshold
        self.iou_threshold_good = self.opt.iou_threshold_for_good_proposal
        # self.label_file_for_tap = self.opt.label_file_for_tap
        self.input_c3d_dir2 = opt.input_c3d_dir2
        with open(self.opt.w1_json) as f:
            self.w1 = json.load(f)

        with open(self.opt.video_json) as f:
            self.data = json.load(f)

        self.use_SOTA_tep = vars(self.opt).get('SOTA_json', None)
        if self.use_SOTA_tep:
            with open(self.opt.SOTA_json) as f:
                self.SOTA_TEP_Poporal = json.load(f)['results']

        self.K = self.opt.K
        self.prop_sample_num = opt.prop_sample_num

        # load json file which contains additional information about dataset
        print('DataLoader loading features file: ', opt.input_c3d_dir2)
        print('DataLoader loading train label file: ', opt.train_label_for_cg)
        print('DataLoader loading val label file: ', opt.val_label_for_cg)

        with open(self.opt.video_data_for_cg) as f:
            self.info = json.load(f)

        print('DataLoader loading video_data_information file: ', opt.video_data_for_cg)

        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        # open the label file
        train_label_h5 = h5py.File(self.opt.train_label_for_cg, 'r', driver='core')
        self.train_label_file = {key: train_label_h5[key].value for
                                 key in train_label_h5.keys()}
        train_label_h5.close()

        val_label_h5 = h5py.File(self.opt.val_label_for_cg, 'r', )

        self.val_label_file = {key: val_label_h5[key].value for key in
                               val_label_h5.keys()}
        val_label_h5.close()

        if vars(self.opt).get('other_features', 0) != 0:
            self.other_feats = self.get_other_feats(self.opt.other_features)

        seq_size = self.train_label_file['labels'].shape

        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)

        # load the index of sentences for all videos
        # end_ix - start_ix is the number of senteces for a video
        self.train_label_start_ix = self.train_label_file['label_start_ix'][:]
        self.train_label_end_ix = self.train_label_file['label_end_ix'][:]
        self.val_label_start_ix = self.val_label_file['label_start_ix'][:]
        self.val_label_end_ix = self.val_label_file['label_end_ix'][:]
        self.val_videos = self.val_label_start_ix.shape[0]
        self.train_videos = self.train_label_start_ix.shape[0]

        print('there are %d videos to be trained' % (self.train_videos))
        print("there are %d videos in validation " % (self.val_videos))
        self.split_ix = {'train': [], 'val': [], 'test': []}
        # separate out indexes for each of the provided splits
        for ix in range(len(self.info['videos'])):
            # if ix % 10 != 0:
            #    continue
            video = self.info['videos'][ix]
            if video['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif video['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif video['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)
        print('assigned %d videos to split train' % len(self.split_ix['train']))
        print('assigned %d videos to split val' % len(self.split_ix['val']))
        print('assigned %d videos to split test' % len(self.split_ix['test']))
        self.train_length = self.train_videos
        self.val_length = self.val_videos
        # self.test_length = len(self.split_ix['test'])
        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split,
                                                        self, (split == 'train') and (opt.shuffle))

        # BlobFetcher(train,self,train)
        # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    # calculate the iou value
    def iou(self, interval, featstamps, return_index=False):
        start_i, end_i = interval[0], interval[1]
        output = 0.0
        gt_index = -1
        for i, (start, end) in enumerate(featstamps):
            start = start - 0.01
            end = end + 0.01
            intersection = max(0, min(end, end_i) - max(start, start_i))
            union = min(max(end, end_i) - min(start, start_i), end - start + end_i - start_i)
            overlap = float(intersection) / (union + 1e-8)
            if overlap >= output:
                output = overlap
                gt_index = i
        if return_index:
            return output, gt_index
        return output

    def event_distance(self, featstamps1, featstamp2):
        s1, e1 = featstamps1
        s2, e2 = featstamp2
        intersection = max(0, min(e1, e2) - max(s1, s2))
        union = min(max(e1, e2) - min(s1, s2), e1 - s1 + e2 - s2)
        d = float(intersection) / (e1 - s1) + float(intersection) / (e2 - s2)
        return d

    # calculat the features for each gt proposal
    def timestamp_to_featstamp(self, timestamp, nfeats, duration):
        start, end = timestamp
        start = max(min(int(round(start / duration * nfeats)), nfeats - 2), 0)
        end = min(max(int(round(end / duration * nfeats)), start + 1), nfeats - 1)
        return start, end

    def featstamp_to_time(self, start_f, end_f, nfeats, duration):
        time_per_feat = duration / nfeats
        start = min(max(0, start_f * time_per_feat), duration - time_per_feat)
        end = max(end_f * time_per_feat, start + time_per_feat)
        return start, end

    def get_SOTA_TEP_label(self, video_id, nfeats):

        duration = self.data[video_id]['duration']
        others = {}
        SOTA_featstamps = None
        SOTA_Prop_score = None
        SOTA_timestamps = None
        if video_id[2:] in self.SOTA_TEP_Poporal.keys():
            SOTA_timestamps = [event['segment'] for event in self.SOTA_TEP_Poporal[video_id[2:]]]
            SOTA_featstamps = [self.timestamp_to_featstamp(x, nfeats, duration) for x in SOTA_timestamps]
            SOTA_Prop_score = [event['score'] for event in self.SOTA_TEP_Poporal[video_id[2:]]]

        # others['SOTA_featstamps'] = SOTA_featstamps
        # others['SOTA_Prop_score'] = SOTA_Prop_score
        return SOTA_featstamps, SOTA_Prop_score, SOTA_timestamps

    def get_vid_data(self, video_id, nfeats):

        # feats = features[video_id]["c3d_features"]

        duration = self.data[video_id]['duration']
        timestamps = self.data[video_id]['timestamps']
        featstamps = [self.timestamp_to_featstamp(x, nfeats, duration) for x in timestamps]

        SOTA_featstamps = None
        SOTA_Prop_score = None
        if self.use_SOTA_tep:
            if video_id[2:] in self.SOTA_TEP_Poporal.keys():
                SOTA_timestamps = [event['segment'] for event in self.SOTA_TEP_Poporal[video_id[2:]]]
                SOTA_featstamps = [self.timestamp_to_featstamp(x, nfeats, duration) for x in SOTA_timestamps]
                SOTA_Prop_score = [event['score'] for event in self.SOTA_TEP_Poporal[video_id[2:]]]

        time_per_feat = duration / nfeats
        nb_prop = len(featstamps)

        iou_scores = np.zeros([nfeats, self.K], dtype='float32')
        gts_index = np.zeros([nfeats, self.K], dtype='float32')
        S_iou_scores = np.zeros([nfeats, nfeats], dtype='float32')
        # gt_captured = []

        tap_masks = np.zeros([nfeats, self.K], dtype='float32')
        S_tap_masks = np.zeros([nfeats, nfeats], dtype='float32')

        for index in range(nfeats):
            tap_masks[index, :min(self.K, index)] = 1

        for t in range(nfeats):
            for k in xrange(self.K):
                if t >= k + 1:
                    iou, gt_index = self.iou([t - k - 1, t], featstamps, return_index=True)
                    iou_scores[t, k] = iou
                    gts_index[t, k] = gt_index
                    S_iou_scores[t - k - 1, t] = iou
                    S_tap_masks[t - k - 1, t] = 1

        others = {}
        others['S_iou_scores'] = S_iou_scores
        others['S_tap_masks'] = S_tap_masks
        others['SOTA_featstamps'] = SOTA_featstamps
        others['SOTA_Prop_score'] = SOTA_Prop_score

        return iou_scores, tap_masks, gts_index, featstamps, others

    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size
        wrapped = False
        infos = []
        prop_captured = []
        data = {}
        for i in range(batch_size):
            # fetch videos,labels,temp_att and some other information
            tmp_c3d, tmp_lda, tmp_att, tap_label, tap_masks, iou_scores, gts_index, tap_masks_good_proposal, train_only, w1, ix, others, tmp_wrapped = \
                self._prefetch_process[split].get()

            ix1 = 0
            ix2 = 0
            video_id = self.info['videos'][ix]['video_id']
            if split == 'train':
                # get the video_id
                # print('train: id:{}'.format(ix))
                ix1 = self.train_label_start_ix[ix]  # label_start_ix starts from 0
                ix2 = self.train_label_end_ix[ix] - 1
            elif split == 'val':
                # print('val: id:{}'.format(ix))
                # video_id = self.info['videos'][ix]['video_id']
                ix1 = self.val_label_start_ix[ix - self.train_length]
                ix2 = self.val_label_end_ix[ix - self.train_length] - 1

            features = tmp_c3d

            ncap = ix2 - ix1 + 1  # number of captions available for this video

            lm_mask_batch = np.zeros([ncap, self.seq_length], dtype='float32')

            # fetch the sequence labels of the video
            if split == 'train':
                lm_label_batch = self.train_label_file["labels"][ix1:ix2 + 1]
            elif split == 'val':
                lm_label_batch = self.val_label_file['labels'][ix1:ix2 + 1]
            if tmp_wrapped:
                wrapped = True

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['videos'][ix]['id']
            infos.append(info_dict)

            # generate lm mask

            lm_label_batch = np.append(lm_label_batch, np.zeros((1, self.seq_length)), axis=0)
            lm_label_batch[:, -1] = 0

            if vars(self.opt).get("dropsent_mode", "nodrop") == "insert":
                for _ix, row in enumerate(lm_mask_batch):
                    nonzeros = (lm_label_batch[_ix] != 0).sum() + 2
                    if nonzeros > 12:
                        if np.random.random() > 0.7:
                            crop_point = np.random.randint(12, nonzeros)
                            lm_label_batch[_ix, (crop_point + 1):] = lm_label_batch[_ix, crop_point:(-1)]
                            lm_label_batch[_ix, crop_point] = 0
                    row[:nonzeros + 1] = 1

            elif vars(self.opt).get("dropsent_mode", "nodrop") == "truncate":
                for _ix, row in enumerate(lm_mask_batch):
                    nonzeros = (lm_label_batch[_ix] != 0).sum() + 2
                    crop_point = nonzeros
                    if nonzeros > 12:
                        if np.random.random() > 0.7:
                            crop_point = np.random.randint(12, nonzeros)
                            lm_label_batch[_ix, crop_point:] = 0
                    row[:min(nonzeros, crop_point + 1)] = 1
            else:
                for _ix, row in enumerate(lm_mask_batch):
                    nonzeros = (lm_label_batch[_ix] != 0).sum() + 2
                    row[:nonzeros] = 1
            lm_label_batch[:, -1] = 0

            sent_len = np.array(list(map(lambda x: (x != 0).sum() + 2, lm_label_batch)), dtype='int')

            # generate index selecting list
            data['fc_feats'] = np.stack(features)
            # data['att_feats'] = np.stack(tmp_att)
            data['att_feats'] = np.array([0]).astype('float32')
            data['lda_feats'] = tmp_lda.astype('float32')
            # data['cg_labels'] = lm_label_batch.astype('int')  # masked  by lm mask
            data['cg_gts'] = gts_index.astype('int')  # masked by tap mask
            # data['cg_masks'] = lm_mask_batch.astype('float32')
            data['cg_sents_len'] = sent_len

            data["tap_labels"] = tap_label.astype('float32')  # masked by tap mask
            data["tap_iou_scores"] = iou_scores  # masked by tap mask

            # data["S_tap_labels"] = others['S_tap_labels'].astype('float32')  # masked by tap mask
            # data['S_tap_masks_for_loss'] = others['S_tap_masks'].astype('float32')

            data["tap_gts_index"] = gts_index.astype('int')  # maked by tap mask
            # data['tap_event_boundary'] = boundary.astype('float32')
            # data['tap_event_desc_score'] = self.desc_scores[video_id].astype('float32')

            data["tap_masks_for_loss"] = tap_masks  # masked by tap mask
            data["tap_masks_for_good_proposal"] = tap_masks_good_proposal.astype('int')  # maksed by tap mask
            tap_gts_for_good_proposal = (tap_masks_good_proposal * (gts_index + 1) - 1).astype('int')

            data['duration'] = self.data[video_id]['duration']
            data['sentences'] = self.data[video_id]['sentences']
            data['gt_featstamps'] = others['gt_featstamps']
            data['gt_timestamps'] = self.data[video_id]['timestamps']

            data['tap_gts_for_good_proposal'] = others['tap_gts_for_good_proposal']

            data['vid'] = video_id
            w1 = 1 - w1 if vars(self.opt).get('reverse_w0', False) else w1
            data['w1'] = w1
            # data['bw1'] = np.array([self.bw1]).astype('float32')
            # data['desc_w1'] = np.array([self.desc_w1]).astype('float32')
            data['bounds'] = {'it_pos_now': self.iterators[split],
                              'it_max': len(self.split_ix[split]),
                              'wrapped': wrapped}
            data['infos'] = infos

            data["proposal_num"] = proposal_num = data['tap_masks_for_good_proposal'].sum().astype('int').tolist()
            data['ix'] = ix

            T, _ = features.shape
            if proposal_num <= 0 or features.shape[0] <= 1:
                return data

            # pdb.set_trace()

            if True:

                featstamp = others['gt_featstamps']
                data['gts_cg_select_list'] = np.array( [i for i,f in enumerate(featstamp)]).astype('int')
                data['gts_ind_select_list'] = np.array([f[1] for f in featstamp]).astype('int')
                data['gts_soi_select_list'] = np.array([[f[0], f[1] + 1] for f in featstamp]).astype('int')
                gt_sentence_batch = []
                for ind in data['gts_cg_select_list']:
                    gt_sentence_batch.append(data['sentences'][ind])
                data['gts_sentences_batch'] = gt_sentence_batch

            data['SOTA_featstamps'] = others['SOTA_featstamps']
            data['SOTA_timestamps'] = others['SOTA_timestamps']
            data['SOTA_Prop_score'] = others['SOTA_Prop_score']

            if self.use_SOTA_tep and (others['SOTA_featstamps'] is not None):
                featstamp = others['SOTA_featstamps']
                for ind, (x, y) in enumerate(featstamp):
                    if y <= x:
                        assert AssertionError
                    assert y > x
                    if y - x >= (self.K + 1):
                        # print('SOTA_TEP: Ding')
                        rand = np.random.randint(0, y - x - (self.K - 1))
                        rand_start = x + rand
                        rand_end = rand_start + (self.K)
                        featstamp[ind] = [rand_start, rand_end]
                data['SOTA_cg_select_list'] = np.array(
                    [tap_gts_for_good_proposal[f[1], f[1] - f[0] - 1] for f in featstamp]).astype('int')
                data['SOTA_ind_select_list'] = np.array([f[1] for f in featstamp]).astype('int')
                data['SOTA_soi_select_list'] = np.array([[f[0], f[1] + 1] for f in featstamp]).astype('int')
                gt_sentence_batch = []
                for ind in data['SOTA_cg_select_list']:
                    gt_sentence_batch.append(data['sentences'][ind])
                data['SOTA_sentences_batch'] = gt_sentence_batch

            # in training phase, random sample proposals
            # if split == 'train':
            if True:
                if self.opt.tap_model == "sst_1stage":
                    data['action_label'] = (others['action_label'] >= 0).astype('int').astype('float32')
                    data['action_label_index'] = others['action_label'].astype('int').astype('float32')

                    data['ind_select_list'] = train_only['ind_select_list']  # sampled
                    data['ind_select_list_eval_1stage'] = train_only['ind_select_list_eval']
                    data['soi_select_list'] = train_only['soi_select_list']
                    data['soi_select_list_eval_1stage'] = train_only['soi_select_list_eval']
                    data['cg_select_list'] = train_only['cg_select_list']
                    data['cg_labels_1stage'] = lm_label_batch[data['cg_select_list']].astype('int')  # sampled
                    data['cg_labels_eval_1stage'] = lm_label_batch.astype('int')
                    data['cg_masks_1stage'] = lm_mask_batch[data['cg_select_list']].astype('float32')  # sampled
                    data['cg_masks_eval_1stage'] = lm_mask_batch.astype('float32')

                    for row, v in enumerate(data['cg_select_list']):
                        assert v >= -1
                        if v == -1:
                            data['cg_masks_1stage'][row] = np.zeros(self.seq_length)


                else:
                    # if True:
                    # assert ncap == tap_gts_for_good_proposal.max() + 1
                    data['ind_select_list'] = train_only['ind_select_list']  # sampled
                    data['soi_select_list'] = train_only['soi_select_list']
                    data['cg_select_list'] = train_only['cg_select_list']
                    data['cg_labels'] = lm_label_batch.astype('int')  # sampled
                    data['cg_masks'] = lm_mask_batch.astype('float32')  # sampled

                sentence_batch = []
                for ind in train_only['cg_select_list']:
                    sentence_batch.append(data['sentences'][ind])
                data['sentences_batch'] = sentence_batch
                data['sampled_ids'] = train_only['sampled_ids']

            if ncap != gts_index.max() + 1:
                pdb.set_trace()
                pass
            assert ncap == gts_index.max() + 1, video_id
        return data


    def get_segment_indics(self, soi_select_list):
        '''
        We devide the relation of event1(host) and event2(customer) into 3 part:
        part1: intersection of event1 and event2(AB)
        part2: the left difference set and the right difference set of event1(A-B)
        part3: the left difference set and the right difference set of event2(B-A)

        :param soi_select_list: list, shape[batch_size,2]
        :return: indics for indicating the index of
        '''
        soi_select_list = [(s, e - 1) for s, e in soi_select_list]
        bs = len(soi_select_list)
        soi_select_list = np.array(soi_select_list)  # (bs,2)

        if len(soi_select_list.shape) <= 1:
            print(soi_select_list.shape, soi_select_list)
        s1, e1 = np.split(np.expand_dims(soi_select_list, 1).repeat(bs, 1), 2, axis=2)  # [bs,bs,1],[bs,bs,1]
        s2, e2 = np.split(np.expand_dims(soi_select_list, 0).repeat(bs, 0), 2, axis=2)  # [bs,bs,1],[bs,bs,1]
        templates = np.concatenate([
            np.maximum(s1, s2), np.minimum(e1, e2),
            s1, np.minimum(e1, s2),
            np.maximum(e2, s1), e1,
            s2, np.minimum(e2, s1),
            np.maximum(e1, s2), e2
        ], 2)  # [bs,bs,10]

        masks1, mask21, mask22, mask31, mask32 = np.minimum(e1, e2) - np.maximum(s1, s2) > 0, \
                                                 np.minimum(e1, s2) - s1 > 0, \
                                                 e1 - np.maximum(e2, s1) > 0, \
                                                 np.minimum(e2, s1) - s2 > 0, \
                                                 e2 - np.maximum(e1, s2) > 0  # [bs,bs,1]

        masks = np.concatenate([masks1, masks1, mask21, mask21, mask22, mask22, mask31, mask31, mask32, mask32],
                               axis=2)  # [bs,bs,10]
        indics = templates * masks
        return indics  # [bs,bs,10]



    def get_boundary(self):
        video_ids = []
        for i in range(len(self.info['videos'])):
            video_ids.append(self.info['videos'][i]['video_id'])

        boundarys = {}
        desc_scores = {}
        featstamps_dict = {}

        p_i = []
        desc_pi = []

        for ii, video_id in enumerate(video_ids):

            feats = self.feats_c3d[video_id]["c3d_features"]
            if 'Diff' in self.opt.tap_model:
                nfeats = int((feats.shape[0] + 1) / 2)
            else:
                nfeats = feats.shape[0]

            duration = self.data[video_id]['duration']
            timestamps = self.data[video_id]['timestamps']
            featstamps = [self.timestamp_to_featstamp(t, nfeats, duration) for t in timestamps]
            featstamps = [(x, min(y, nfeats - 1)) for (x, y) in featstamps]

            sent_len = [len(sent.split()) for sent in self.data[video_id]['sentences']]
            boundary = np.zeros([nfeats, 2])
            desc_score = np.zeros([nfeats])
            for i, (x, y) in enumerate(featstamps):
                # pdb.set_trace()
                # print('video_id:{},len:{},featstamp:{}'.format(video_id,nfeats,(x,y)))
                boundary[y, 0] = 1
                boundary[x, 0] = 1
                desc_score[x:y + 1] = desc_score[x:y + 1] + sent_len[i] * 1.0 / (y - x + 1)
            boundary[:, 1] = 1 - boundary[:, 0]

            # desc_score[:,0] = 1.0 * desc_score[:,0] / np.sum(desc_score[:,0])
            desc_score = 1.0 * desc_score

            # desc_score[:,1] = 1 - desc_score[:,0]

            boundarys[video_id] = boundary
            desc_scores[video_id] = desc_score
            featstamps_dict[video_id] = featstamps

            p_i.append(len(timestamps) * 2.0 / nfeats)
            desc_pi.append(1.0 * np.sum(desc_score != 0) / nfeats)

        return boundarys, np.array(p_i).mean(), desc_scores, np.array(desc_pi).mean(), featstamps_dict



    def get_shuffle_list(self, tap_gts_for_good_proposal, gt_featstamps, method='random'):
        if method == 'random':
            tap_indices_selecting_list = []
            lm_indices_selecting_list = []
            soi_indices_selecting_list = []
            for i, row in enumerate(tap_gts_for_good_proposal):
                for j, index in enumerate(row):
                    if not index == -1:
                        tap_indices_selecting_list.append(i)
                        lm_indices_selecting_list.append(index)
                        soi_indices_selecting_list.append([i - j, i + 1])
            proposal_num = len(tap_indices_selecting_list)
            sampled_ids = np.array(range(proposal_num)).astype('int')
            np.random.shuffle(sampled_ids)
            sampled_ids = sampled_ids[:min(proposal_num, self.prop_sample_num)]

            tap_indices_selecting_list = np.array(tap_indices_selecting_list)
            lm_indices_selecting_list = np.array(lm_indices_selecting_list)
            soi_indices_selecting_list = np.array(soi_indices_selecting_list)

            # tap_sample_list = np.array(tap_indices_selecting_list)[sampled_ids]
            # lm_sample_list = np.array(lm_indices_selecting_list)[sampled_ids]
            # soi_sample_list = np.array(soi_indices_selecting_list)[sampled_ids]

            return tap_indices_selecting_list, lm_indices_selecting_list, soi_indices_selecting_list, sampled_ids

        elif method == "1stage":
            proposal_num, K = tap_gts_for_good_proposal.shape

            # actionness = np.zeros(proposal_num)
            # for gt_stamp in gt_featstamps:
            #     actionness[gt_stamp[0]: gt_stamp[1] + 1] = 1


            action_label = np.zeros((proposal_num,K+1)) - 1
            sampled_ids = np.arange(proposal_num).astype('int')
            np.random.shuffle(sampled_ids)
            sampled_ids = sampled_ids[:min(proposal_num, self.prop_sample_num)]

            lm_indices_selecting_list = []
            soi_indices_selecting_list = []
            for i, row in enumerate(tap_gts_for_good_proposal):

                sent_ids = np.zeros(1 + np.max(tap_gts_for_good_proposal))
                for j, index in enumerate(row):
                    if index >= 0:
                        sent_ids[index] += 1
                if sent_ids.sum() > 0:
                    sent_id = sent_ids.argmax()
                else:
                    sent_id = -1

                lm_indices_selecting_list.append(sent_id)
                soi_indices_selecting_list.append([max(0, i - K), i + 1])


                if sent_id>=0:
                    actionness = np.zeros(proposal_num)-1
                    actionness[gt_featstamps[sent_id][0]: gt_featstamps[sent_id][1] + 1] = sent_id
                    selected = actionness[max(0, i - K): i + 1]
                    action_label[i, :len(selected)] = selected


            tap_indices_selecting_list = np.arange(proposal_num)
            lm_indices_selecting_list = np.array(lm_indices_selecting_list)
            soi_indices_selecting_list = np.array(soi_indices_selecting_list)

            return tap_indices_selecting_list, lm_indices_selecting_list, soi_indices_selecting_list, sampled_ids, action_label

        elif method == 'avg':  # 此代码未测试，0726bywangt,
            ncap = tap_gts_for_good_proposal.max() + 1
            prop_per_cap = int(self.prop_sample_num / ncap)
            whole_props_per_cap = [[cap, (tap_gts_for_good_proposal == cap).sum()] for cap in range(ncap)]
            sorted(whole_props_per_cap, key=lambda x: x[1], reverse=False)
            sampled_ids = []
            for cap_id, prop_num in whole_props_per_cap:
                bg_inds = zip(np.where(tap_gts_for_good_proposal == cap_id))
                if len(bg_inds) > prop_per_cap:
                    bg_inds = np.random.choice(bg_inds, size=prop_per_cap)
                sampled_ids.extend(bg_inds)
            tap_sample_list = zip(*sampled_ids)[0]
            lm_sample_list = tap_gts_for_good_proposal[sampled_ids]
            soi_sample_list = [[t - k, t + 1] for t, k in sampled_ids]
        else:
            raise ValueError('sample method wrong')



    def get_diff_tap_label(self, tap_label):
        return

    '''
    def get_topNneighbor_feature(self, soi_list, c3d_feature, lda_feats=None, topN=36):
        dist_topn = self.get_topNneighbor(soi_list, N=topN)
        soi_feature = []
        local_ind_for_tap = []
        for topN in dist_topn:
            local_soi_list = soi_list[topN]
            pooled = [c3d_feature[soi[0]:soi[1]].mean(0) for soi in local_soi_list]  # local_N * vid_dim
            index = [soi[1] - 1 for soi in local_soi_list]  # local_N * tap_hidden_dim
            # tmp=np.array(tmp)
            # local_soi_feature=np.concatenate((pooled,tmp),1) #local_N * (vid_dim)
            soi_feature.append(pooled)
            local_ind_for_tap.append(index)
        soi_feature = np.array(soi_feature)  # event_num * local_N * (vid_dim)
        local_ind_for_tap = np.array(local_ind_for_tap)
        return soi_feature, local_ind_for_tap
    
    def get_topNneighbor(self, soi_list, N=36):
        proposal_num = len(soi_list)
        dist_array = np.zeros((proposal_num, proposal_num))
    
        for n in range(proposal_num):
            for k in range(n + 1):
                dist_array[n, k] = self.event_distance(soi_list[n], soi_list[k])
                dist_array[k, n] = dist_array[n, k]
        sort_ind = np.argsort(-dist_array, axis=1)
        topN = sort_ind[:, :N]
        if False:
            for n, i in enumerate(topN):
                print('event:', n, soi_list[n])
                print(soi_list[i])
        return topN
    '''

    # It's not coherent to make DataLoader a subclass of Dataset,
    # but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according
    # the index. However, it's minimum change to switch to pytorch data loading

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        return self.get_data(ix)

    def __len__(self):
        return len(self.info['videos'])

    def get_v_GwIHO7HpGkY(self):
        while True:
            data = self.get_batch('train')
            if data['vid'] == 'v_GwIHO7HpGkY':
                return data


class ArraySampler(data.sampler.SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name,
        caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases:
        1. not hasattr(self, 'split_loader'): Resume from previous training.
        Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in
         the get_minibatch_inds already.
        """
        # batch_size is 0, the merge is done in DataLoader class
        if self.if_shuffle:
            random.shuffle(self.dataloader.split_ix[self.split])
        sampler = ArraySampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:])
        # print(self.split, len(sampler))
        self.split_loader = iter(
            data.DataLoader(dataset=self.dataloader,
                            batch_size=1,
                            sampler=sampler,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=self.dataloader.opt.nthreads,
                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False
        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            wrapped = True

        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()
        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()
        assert tmp[10] == ix, "ix not equal"
        return tmp + [wrapped]


class miniloader(object):
    def __init__(self, path):
        self.iterators = []
        self.split_ix = []
        self.saved_path = path
        self.minidata = cPickle.load(open(self.saved_path))
        self.traindata = iter(self.minidata['train'])
        self.valdata = iter(self.minidata['val'])
        self.vocab = self.minidata['vocab']
        self.vocab_size = len(self.vocab)
        self.seq_length = 32

    def get_batch(self, mode):
        if mode == 'train':
            return self.traindata.next()
        elif mode == 'val':
            return self.valdata.next()

    def get_vocab(self):
        return self.vocab

    def featstamp_to_time(self, start_f, end_f, nfeats, duration):
        time_per_feat = duration / nfeats
        start = min(max(0, start_f * time_per_feat), duration - 1)
        end = max(end_f * time_per_feat, start + 1)
        return start, end

    def reset_iterator(self, mode):
        return

    def __len__(self):
        return 200

def get_and_save_w1_json(path):
    import opts
    opt = opts.parse_opts()
    opt.K = 256
    opt.iou_threshold = 0.5

    opt.input_c3d_dir2 = "data/c3d_npy_s32"
    loader = DataLoader(opt)
    propo_pos_examples = []
    for i in range(10009):
        d = loader.get_batch('train')
        print(i, d['vid'])
        iou_scores = d["tap_iou_scores"]
        tap_labels = (iou_scores > opt.iou_threshold)

        nfeats = d['fc_feats'].shape[0]
        propo_pos_examples += [np.sum(tap_labels, axis=0) * 1. / nfeats]
    w1 = 1 - np.array(propo_pos_examples).mean(axis=0)  # train and val dataset
    json.dump(w1.tolist(), open(path, 'w'))

def plot_matrix(m, path):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    # plt.imshow(m,cmap=plt.cm.gray)
    # plt.show()
    plt.imshow(m, cmap=plt.cm.hot, vmin=np.amin(m), vmax=np.amax(m))
    plt.colorbar()
    plt.savefig(path)


if __name__ == '__main__':
    # get_and_save_w1_json('data/w1_256_c3d32_iou0.5.json')
    import opts
    import pdb

    opt = opts.parse_opts()
    # opt.input_c3d_dir = 'data/sub_activitynet_v1-3.c3d_sampling_per32frame.hdf5'
    # opt.prop_sample_num = 200
    # opt.lm_epochs = 30
    # opt.use_att = False
    # opt.nthreads = 8
    # opt.fusion_model = 'MA3'
    # print(opt.nthreads)
    # opt.label_file_for_tap = 'data/label_file_for_tap_128_by_wt_4.0.h5'
    # tap_opt.label_file_for_tap = 'data/label_file_for_tap_64_by_testing.h5'
    opt.tap_model = 'sst_1stage'
    opt.use_2stream_feature = 0
    opt.input_twostream_dir = '/data/huichengzheng/wangteng/bsn/data/activitynet_feature_cuhk/full_feature'
    loader = DataLoader(opt)

    for i in range(100):
        tmp = loader.get_batch('train')
    pass

''''
    import time

    #opt.use_att = True
    s = time.time()
    r_sum = []
    data_q = {'train': [],
              'val': [],
              'vocab': {}}

    #for i in range(1000):
    #    print(i)
    #    tmp = loader.get_batch('test')
        # pdb.set_trace()
        # data_q['train'].append(tmp)
        # for i in range(6000):
        #    data_ = loader.get_batch('val')
    for i in range(20):
        # print(i)
        tmp = loader.get_batch('val')
        pdb.set_trace()
        # pdb.set_trace()
        # data_q['val'].append(tmp)
    data_q['vocab'] = loader.get_vocab()

    print(time.time() - s)


    p = '/home/wangteng/extdisk/project/DenseVideoCaptioning.pytorch/data/mini_dataset0910.pkl'
    import cPickle

    cPickle.dump(data_q, open(p, 'w'))

'''
