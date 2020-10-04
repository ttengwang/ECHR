from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import scripts.wt_preprocessing as wt

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
    loss_evals = 1e-8
    predictions = []
    tap_cg_pred = {}
    iter = 0
    bad_vid_num = 0

    time_consumption = {}
    with torch.set_grad_enabled(False):
        while True:
            data = loader.get_batch(split)
            #sign = get_interest(data['vid'])
            #if not sign:
            #    continue

            # data = utils.get_dummy_data(tap_opt,eval_kwargs)
            n = n + 1
            # n = n + loader.batch_size
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

            torch.cuda.synchronize()
            t0 = time.time()
            tap_feats, pred_proposals = tap_model(c3d_feats)

            torch.cuda.synchronize()
            t1 = time.time()
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

            t2 = time.time()

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
                    torch.cuda.synchronize()
                    t3 = time.time()

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
            time_consumption[iter] = {'tep': t1 - t0, 'cg': t3-t2, 'postprocess': t2-t1}
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

    json.dump(time_consumption, open(json_path+'.time_consumption.json', 'w'))

    sys.path.append('external_tool/densevid_eval')
    sys.path.append('external_tool/densevid_eval/coco-caption')

    score = {'ARAN':0}
    if lang_eval:
        from evaluate import eval_score

        sample_score = eval_score(json_path, flag_eval_what == 'tap', eval_kwargs['val_all_metrics'])
        for key in sample_score.keys():
            score[key] = np.array(sample_score[key])
        print('vilid vid num:{}, bad_num:{}'.format((eval_kwargs['num_vids_eval'] - bad_vid_num), bad_vid_num))

    if flag_eval_what=='tap':
        import external_tool.eval_ARAN.get_proposal_performance as eval_score_tap
        eval_tap_opt = {}
        eval_tap_opt['ground_truth_filename']='/data/huichengzheng/wangteng/dvc2_pytorch04/data/captiondata/val_forARAN.json'
        eval_tap_opt['proposal_filename']=json_path
        score['ARAN'] = eval_score_tap.main(**eval_tap_opt)

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


'''
def S_gettop1000(pred_proposals, tap_masks, lm_gts, duration, featstamp_to_time, val_score_thres=0, topN=1000):
    nfeats, K = pred_proposals.shape
    pred_proposals = pred_proposals * torch.FloatTensor(tap_masks).cuda()
    sort, _ = pred_proposals.view(-1).sort()
    score_threshold = sort[-min(len(sort), topN)]
    good_proposals = pred_proposals >= max(score_threshold, val_score_thres)
    index_select_list = []
    lm_select_list = []
    timestamp_list = []
    featstamp_list = []
    confidence = []
    for n in range(nfeats):
        for k in range(K):
            if tap_masks[n,k]==1 and good_proposals[n, k] == 1:
                index_select_list.append(k)
                if len(lm_gts):  # if lm_gts is not none
                    lm_select_list.append(lm_gts[k, k-n-1])
                timestamp = featstamp_to_time(n, k, nfeats, duration)
                timestamp_list.append(timestamp)
                featstamp_list.append([n, k + 1])
                confidence.append(pred_proposals[n, k])

    return index_select_list, featstamp_list, lm_select_list, timestamp_list, confidence

'''


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


def get_interest(vid):
    interest = ['v_hDPLy21Yyuk','v_CSDApI2nHPU', 'v_6F9C3dIU4kU','v_A8xThM3onkc']
    cand = 'v_yACg55C3IlM v_ZcgahXg_ELw v_gLfvk2SSj1c v_pMDFkrK0KRc v_tyjUDi3uLd0 v_A7ER02-zr54 v_sG3JpMuXFnU v_sIYRsGZm2XY v_8TGG-FZx0cc v_Ce0t7gfJl5w v_iosb2TdQ7yY v_EGLJPCJnG64 v_oKauZV0DHHk v_5hXH-TorJ6M v_mio5dnRbo4w v_Sd08rsPTroE v_JQpx7CcTstU v_LsK452h29ng v_WZeMQ-5dFlM v_g_Cz69Q5bKM v__b_9BQvJ_v4 v_M1-G6KEhY-M v_kyafh7Ownao v_291szrilAVE v__8aVDfNQtq0 v_BodF651KcIg v_vdTisVMhW7I v_s_VFaQTlskE v_Rai5nKbB6wU v_furUOKw0Qzs v_1ioKX0iuico v_onBAyGhqubg v_fZQS02Ypca4 v_UnOzWl0EGCA v_E7C91KoML-o v_r8AXq1Q5bn0 v_TxYZLJQOHvY v_OzXD3WO6jrs v_ANwaFSIHdW0 v_4x3dgSgXQ38 v_mSonugqhYuE v_KEXm-3H6eTg v_Vtsv9iPHDqg v_e4XYZAs7tcs v_FLZPaPf027E v_R6kXT4Spiwo v_uF9othvTXn8 v_Ff8QLpH5T1c v_SKtUq_1cOSs v_1XtjXqqPvyQ v_ulopyhvgyQg v_EZKrOWEKX_Q v__15t4WTR19s v_zmmiX3_TJ84 v_ss6XN-JP_x8 v_-F7QWQA8Eh8 v_ku65ME0vW8s v_O0KUnuhLwj0 v_6hjRnngC73o v_37gHYr2uDZo v_Ta_Kf0dCd3U v_hvy_V1EWKEI v_nKa1e_CpvoY v_sWQ65uwxXbA v_BJ9r8_JnG0k v_l9o9R7UcPuc v_w3N0Pyz2-m0 v_jmxzDxfSbZM v_GLL1vOrV5Qo v_rZu5ZJmAlbI v_DepG0r3JiV4 v_TspdPLMqTx0 v_FEqLmpNzxdg v_qy-LbstiMYg v_9VWoQpg9wqE v_pnEYhDVXVJ0 v_FQkvwPpDomw v_5Foo5NSjEXQ v_HM_rHjh-wqQ v_7ZbH4vHTmVs v_SqIVJrXxO3g v_Gfsk28SzgXk v_JoQywfQ6B-8 v_e6J_ygZ779A v_hBjVRKwCUNA v_MmOQhq95Z_g v_b8eqn-GTdcc v_9iJ8snVY2s0 v_rWdXyKZnL2U v_obUkL-Ya8dE v_3L0MnbQkLWM v_ah3tGziTbds v_DzxPreFrmFE v_2DwBXRhtX4s v_49drGj3JUg4 v_pXcFBfv5Sf4'
    interest = cand.split(' ')
    #interest = ['v_A8xThM3onkc']
    if vid in interest:
        return True
    else:
        return False

def relation_analyse(vid, vid_info):
    interest = ['v_hDPLy21Yyuk', 'v_CSDApI2nHPU', 'v_6F9C3dIU4kU', 'v_A8xThM3onkc']
    # interest = ['v_A8xThM3onkc']
    cand = 'v_yACg55C3IlM v_ZcgahXg_ELw v_gLfvk2SSj1c v_pMDFkrK0KRc v_tyjUDi3uLd0 v_A7ER02-zr54 v_sG3JpMuXFnU v_sIYRsGZm2XY v_8TGG-FZx0cc v_Ce0t7gfJl5w v_iosb2TdQ7yY v_EGLJPCJnG64 v_oKauZV0DHHk v_5hXH-TorJ6M v_mio5dnRbo4w v_Sd08rsPTroE v_JQpx7CcTstU v_LsK452h29ng v_WZeMQ-5dFlM v_g_Cz69Q5bKM v__b_9BQvJ_v4 v_M1-G6KEhY-M v_kyafh7Ownao v_291szrilAVE v__8aVDfNQtq0 v_BodF651KcIg v_vdTisVMhW7I v_s_VFaQTlskE v_Rai5nKbB6wU v_furUOKw0Qzs v_1ioKX0iuico v_onBAyGhqubg v_fZQS02Ypca4 v_UnOzWl0EGCA v_E7C91KoML-o v_r8AXq1Q5bn0 v_TxYZLJQOHvY v_OzXD3WO6jrs v_ANwaFSIHdW0 v_4x3dgSgXQ38 v_mSonugqhYuE v_KEXm-3H6eTg v_Vtsv9iPHDqg v_e4XYZAs7tcs v_FLZPaPf027E v_R6kXT4Spiwo v_uF9othvTXn8 v_Ff8QLpH5T1c v_SKtUq_1cOSs v_1XtjXqqPvyQ v_ulopyhvgyQg v_EZKrOWEKX_Q v__15t4WTR19s v_zmmiX3_TJ84 v_ss6XN-JP_x8 v_-F7QWQA8Eh8 v_ku65ME0vW8s v_O0KUnuhLwj0 v_6hjRnngC73o v_37gHYr2uDZo v_Ta_Kf0dCd3U v_hvy_V1EWKEI v_nKa1e_CpvoY v_sWQ65uwxXbA v_BJ9r8_JnG0k v_l9o9R7UcPuc v_w3N0Pyz2-m0 v_jmxzDxfSbZM v_GLL1vOrV5Qo v_rZu5ZJmAlbI v_DepG0r3JiV4 v_TspdPLMqTx0 v_FEqLmpNzxdg v_qy-LbstiMYg v_9VWoQpg9wqE v_pnEYhDVXVJ0 v_FQkvwPpDomw v_5Foo5NSjEXQ v_HM_rHjh-wqQ v_7ZbH4vHTmVs v_SqIVJrXxO3g v_Gfsk28SzgXk v_JoQywfQ6B-8 v_e6J_ygZ779A v_hBjVRKwCUNA v_MmOQhq95Z_g v_b8eqn-GTdcc v_9iJ8snVY2s0 v_rWdXyKZnL2U v_obUkL-Ya8dE v_3L0MnbQkLWM v_ah3tGziTbds v_DzxPreFrmFE v_2DwBXRhtX4s v_49drGj3JUg4 v_pXcFBfv5Sf4'
    interest = cand.split(' ')

    if vid in interest:
        attention_map = np.load('figs/relation_softmax.npy')
        n,m=attention_map.shape
        for p in vid_info:
            print(p['sentence'])

        pdb.set_trace()
        '''
        for i in range(n):
            idx = np.argsort(-1 * attention_map[i])
            good_id = idx[:10]
            print(good_id)
            for id in good_id:
                print('relation of id{} and id{}'.format(i,id))
                print(vid_info[id])
            pdb.set_trace()
        '''
