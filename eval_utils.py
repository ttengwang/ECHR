from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np
import json
import time
import sys
import misc.utils as utils

def eval_split(models, crits, loader, json_path, eval_kwargs={}, flag_eval_what='tap', debug=False):
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    val_score_thres = eval_kwargs.get('val_score_thres', 0)
    nms_threshold = eval_kwargs.get('nms_threshold', 0)
    is_reranking = eval_kwargs.get('reranking', False)
    print('is_reranking', is_reranking)
    topN = eval_kwargs.get('topN', 1000)
    get_eval_loss = eval_kwargs.get('get_eval_loss', 1)

    tap_model, cg_model = models
    tap_crit, cg_crit = crits

    for model in models:
        model.eval()
    loader.reset_iterator(split)

    n = 0
    loss_sum = [0, 0, 0, 0, 0]
    tap_cg_pred = {}
    iter = 0
    bad_vid_num = 0

    with torch.set_grad_enabled(False):
        while True:
            data = loader.get_batch(split)
            n = n + 1
            if iter % int(len(loader) / 100) == 0:
                print('generating result.json:{:.3f}%'.format(100 * iter / len(loader)))

            if data.get('proposal_num', 1) == 0 or data['fc_feats'].shape[0] <= 1:
                continue

            tmp = [
                   data['fc_feats'],
                   data['att_feats'],
                   data['lda_feats']]

            tmp = [Variable(torch.from_numpy(_)).cuda() for _ in tmp]
            c3d_feats, att_feats, lda_feats = tmp

            tap_feats, pred_proposals = tap_model(c3d_feats)

            # select top score 1000 proposals
            cg_gts = data['cg_gts'] if data.get('cg_labels', None) is not None else []

            if flag_eval_what == 'cg':
                ind_select_list = data['gts_ind_select_list']
                soi_select_list = data['gts_soi_select_list']
                cg_select_list = data['gts_cg_select_list']
                #good_time_stamps = [loader.featstamp_to_time(s, e, len(data['fc_feats']), data['duration']) for (s, e) in soi_select_list]
                good_time_stamps = data['gt_timestamps']
                tap_prob = [1] * len(ind_select_list)

            elif flag_eval_what == 'cg_extend':
                ind_select_list, soi_select_list, cg_select_list, sampled_ids, = data['ind_select_list'], data[
                    'soi_select_list'], data['cg_select_list'], data['sampled_ids']
                good_time_stamps = [loader.featstamp_to_time(s, e, len(data['fc_feats']), data['duration']) for (s, e)
                                    in
                                    soi_select_list]
                tap_prob = [1] * len(ind_select_list)

            elif flag_eval_what == 'SOTA_TEP':
                if data['SOTA_Prop_score'] is None:
                    print('bad video for SOTA_TEP, vid:{}'.format(data['vid']))
                    bad_vid_num += 1
                    continue
                _ind_select_list, _soi_select_list, _cg_select_list = data['SOTA_ind_select_list'], data[
                    'SOTA_soi_select_list'], data['SOTA_cg_select_list']
                #_good_time_stamps = [loader.featstamp_to_time(s, e, len(data['fc_feats']), data['duration']) for (s, e)                                     in _soi_select_list]
                _good_time_stamps = data['SOTA_timestamps']
                _tap_prob = data['SOTA_Prop_score']
                ind_select_list, soi_select_list, cg_select_list, good_time_stamps, tap_prob = [], [], [], [], []

                if nms_threshold > 0:
                    _,_,pick = gettopN_nms(_good_time_stamps, _tap_prob, _tap_prob, nms_overlap=nms_threshold, topN=1000)
                else:
                    pick = list(range(len(_tap_prob)))

                for i, p_scpre in enumerate(_tap_prob):
                    if i not in pick:
                        continue
                    if p_scpre >= val_score_thres:
                        ind_select_list.append(_ind_select_list[i])
                        soi_select_list.append(_soi_select_list[i])
                        if len(_cg_select_list):
                            cg_select_list.append(_cg_select_list[i])
                        good_time_stamps.append(_good_time_stamps[i])
                        tap_prob.append(_tap_prob[i])
                    if len(ind_select_list) >= topN:
                        break


            elif flag_eval_what=='cg' or flag_eval_what=='tap_cg' or flag_eval_what=='tap':
                if nms_threshold != 0:
                    ind_select_list, soi_select_list, cg_select_list, good_time_stamps, tap_prob = \
                        gettop1000_nms(pred_proposals.data, data['tap_masks_for_loss'], cg_gts, data['duration'],
                                       loader.featstamp_to_time, overlap=nms_threshold, topN=topN)
                else:
                    ind_select_list, soi_select_list, cg_select_list, good_time_stamps, tap_prob = \
                        gettop1000(pred_proposals.data, data['tap_masks_for_loss'], cg_gts, data['duration'],
                                   loader.featstamp_to_time, val_score_thres=val_score_thres, topN=topN)

            else:
                assert 1==0


            if (len(cg_select_list) == 0) and (split != 'test'):
                sents = []
            else:
                if flag_eval_what == 'tap':
                    sents = [0] * len(ind_select_list)
                    cg_prob = [0] * len(ind_select_list)
                    cg_score = [0] * len(ind_select_list)
                else:
                    seq, cg_prob = cg_model(tap_feats, c3d_feats, lda_feats, [], ind_select_list, soi_select_list,
                                            mode='eval')
                    if len(seq) == 0:
                        sents = []
                    else:
                        cg_score = cg_prob.sum(1).cpu().numpy().astype('float')
                        # cg_prob = np.round(cg_prob, 3).tolist()
                        sents = utils.decode_sequence(loader.get_vocab(), seq)  # [proposal_num , max_sent_len]

            # get val_loss
            if get_eval_loss and tap_crit and (data.get('cg_labels', None) is not None) and len(cg_select_list) and (split != 'test'):
                tmp = [data['tap_labels'], data['tap_masks_for_loss'], data['cg_labels'][cg_select_list],
                       data['cg_masks'][cg_select_list], data['w1']]
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
                tap_labels, tap_masks_for_loss, cg_labels, cg_masks, w1 = tmp
                tap_loss = tap_crit(pred_proposals, tap_masks_for_loss, tap_labels, w1)

                loss_sum[0] = loss_sum[0] + tap_loss.item()
                if flag_eval_what != 'tap':
                    pred_captions = cg_model(tap_feats, c3d_feats, lda_feats, cg_labels, ind_select_list,
                                             soi_select_list,
                                             mode='train')

                    cg_loss = cg_crit(pred_captions, cg_labels[:, 1:], cg_masks[:, 1:])
                    loss_sum[1] = loss_sum[1] + cg_loss.item()
                    total_loss = eval_kwargs['lambda1'] * tap_loss + eval_kwargs['lambda2'] * cg_loss
                    loss_sum[2] = loss_sum[2] + total_loss.item()

            vid_info = []
            for i, sent in enumerate(sents):
                proposal_info = {}
                proposal_info['sentence'] = sent
                proposal_info['timestamp'] = good_time_stamps[i]
                # proposal_info['cg_prob'] = cg_prob[i]
                proposal_info['sentence_confidence'] = cg_score[i]
                proposal_info['proposal_score'] = tap_prob[i]
                proposal_info['re_score'] = 10 * tap_prob[i] + cg_score[i]
                proposal_info['num'] = [i, len(sents)]
                vid_info.append(proposal_info)

            if len(vid_info) != 0:
                if is_reranking:
                    vid_info = reranking(vid_info)
                tap_cg_pred[data['vid']] = vid_info

            if data['bounds']['wrapped']:
                loader.reset_iterator(split)
                break

            if iter == eval_kwargs['num_vids_eval']:
                loader.reset_iterator(split)
                break

            '''
            if iter%500==0:
                pred2json = {'results': tap_lm_pred,
                             'version': "VERSION 1.0",
                             "external_data":
                                 {
                                     "used": True,
                                     "details": "First fully-connected layer from VGG-16 pre-trained on ILSVRC-2012 training set"
                                 }
                             }
                with open(json_path+'iter{}'.format(iter), 'w') as f:
                    json.dump(pred2json, f)
            '''
            iter += 1
            #relation_analyse(data['vid'], vid_info)
            # torch.cuda.empty_cache()

    pred2json = {
        'results': tap_cg_pred,
                 'version': "VERSION 1.0",
                 "external_data":
                     {
                         "used": True,
                         "details": "First fully-connected layer from VGG-16 pre-trained on ILSVRC-2012 training set"
                     }
                 }

    with open(json_path, 'w') as f:
        json.dump(pred2json, f)

    sys.path.append('external_tool/densevid_eval')
    sys.path.append('external_tool/densevid_eval/coco-caption')

    score = {'ARAN':0}
    if lang_eval:
        from evaluate import eval_score
        sample_score = eval_score(json_path, flag_eval_what == 'tap', eval_kwargs['val_all_metrics'])
        for key in sample_score.keys():
            score[key] = np.array(sample_score[key])
        print('vilid vid num:{}, bad_num:{}'.format((eval_kwargs['num_vids_eval'] - bad_vid_num), bad_vid_num))

    # Switch back to training mode
    for model in models:
        model.train()

    return tap_cg_pred, score, np.array(loss_sum) / iter


def gettopN_nms(props, prop_scores, sent_score, nms_overlap=0.999, topN=1000):

    props = np.array(props)
    prop_scores = np.array(prop_scores)
    sent_score = np.array(sent_score)

    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(prop_scores)
    area = (t2 - t1 + 1e-3).astype(float)
    pick = []
    while (len(ind) > 0) and (len(pick) < topN):
        i = ind[-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1e-3)
        o = wh / (area[i] + area[ind] - wh)
        same_id = ind[np.nonzero(o >= nms_overlap)[0]]
        max_sent_id = np.argmax(sent_score[same_id])
        pick_ind = same_id[max_sent_id]
        pick.append(pick_ind)
        ind = ind[np.nonzero(o <= nms_overlap)[0]]

    nms_props, nms_scores = props[pick, :], prop_scores[pick]

    # print('Number of proposals(after nms {}):'.format(overlap), len(nms_props))
    return nms_props, nms_scores,pick


def gettop1000(pred_proposals, tap_masks, cg_gts, duration, featstamp_to_time, val_score_thres=0, topN=1000):

    nfeats, K = pred_proposals.shape
    if hasattr(pred_proposals,'new_tensor'):
        pred_proposals = pred_proposals.cpu().numpy()
    pred_proposals = pred_proposals * tap_masks
    sort = np.sort(pred_proposals.reshape(-1))
    score_threshold = sort[-min(len(sort), topN)]


    good_proposals = (pred_proposals >= max(score_threshold, val_score_thres))
    index_select_list = []
    cg_select_list = []
    timestamp_list = []
    featstamp_list = []
    confidence = []

    for n in range(nfeats):
        for k in range(K):
            if n >= k and good_proposals[n, k] == 1:
                index_select_list.append(n)
                if len(cg_gts):  # if cg_gts is not none
                    cg_select_list.append(cg_gts[n, k])
                timestamp = featstamp_to_time(n - k, n + 1, nfeats, duration)
                timestamp_list.append(timestamp)
                featstamp_list.append([n - k, n + 1])
                confidence.append(pred_proposals[n, k].item())

    return index_select_list, featstamp_list, cg_select_list, timestamp_list, confidence


def gettop1000_nms(pred_proposals, tap_masks, cg_gts, duration, featstamp_to_time, overlap=0.8, topN=1000):
    props = []
    scores = []
    nfeats, K = pred_proposals.shape
    if hasattr(pred_proposals,'new_tensor'):
        pred_proposals = pred_proposals.cpu().numpy()

    # print('Number of proposals: before nms:', nfeats * K)
    prop_gts = []
    for n in range(nfeats):
        for k in range(min(n, K)):
            props.append([n - k, n + 1])
            if len(cg_gts):
                prop_gts.append(cg_gts[n, k])
            scores.append(pred_proposals[n, k].item())

    props = np.array(props)
    prop_gts = np.array(prop_gts)
    scores = np.array(scores)

    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1 + 1).astype(float)
    pick = []
    while (len(ind) > 0) and (len(pick) < topN):
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]

    if len(cg_gts):
        prop_gts = prop_gts[pick]
    index_select_list = nms_props[:, 1] - 1
    timestamp_list = [featstamp_to_time(s, e, nfeats, duration) for (s, e) in nms_props]
    # print('Number of proposals(after nms {}):'.format(overlap), len(nms_props))
    return index_select_list, nms_props, prop_gts, timestamp_list, nms_scores


def reranking(vid_info):
    re_score = []
    for video in vid_info:
        re_score.append(video['re_score'])
    tmp = np.array(re_score)
    tmp.sort()
    score_threshold = tmp[-min(len(tmp), 10)]
    reranking_info = []
    for video in vid_info:
        if video['re_score'] >= score_threshold:
            reranking_info.append(video)
    return reranking_info