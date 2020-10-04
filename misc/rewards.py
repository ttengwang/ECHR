from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch

import sys,pdb

#sys.path.append("cider")
#from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append("external_tool/densevid_eval/coco-caption")
sys.path.append("cider")
from pycocoevalcap.meteor.meteor_22 import Meteor
from pyciderevalcap.ciderD.ciderD import CiderD as Cider

Meteor_scorer = None
Cider_scorer = None

# CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global Meteor_scorer
    global Cider_scorer
    Meteor_scorer = Meteor_scorer or Meteor()
    Cider_scorer = Cider_scorer or Cider(df='coco-val')

def remove_nonascii(text):
    PUNCTUATIONS = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";", '\n', '\t', '\r']
    for p in PUNCTUATIONS:
        text = text.replace(p,' ')
    text = text.replace('  ',' ')
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, vid_info, gen_result, vocab, opt):
    vid, sentences_batch= vid_info

    start = time.time()
    batch_size,sent_len = gen_result.size()  # batch_size = sample_size * seq_per_img

    # get greedy decoding baseline
    gen_result = utils.decode_sequence(vocab, gen_result)
    greedy_res = utils.decode_sequence(vocab, greedy_res)

    # pdb.set_trace()
    res = OrderedDict()

    #gen_result = gen_result.data.cpu().numpy()
    #greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        #res[i] = [array_to_str(gen_result[i])]
        res[i] = [remove_nonascii(gen_result[i])]
    for i in range(batch_size):
        #res[batch_size + i] = [array_to_str(greedy_res[i])]
        res[batch_size + i] = [remove_nonascii(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(sentences_batch)):
        gts[i] = [remove_nonascii(sentences_batch[i])]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]

    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i%batch_size] for i in range(2 * batch_size)}

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]  # for cider
    gts = {i: gts[i % batch_size] for i in range(batch_size)}

    #pdb.set_trace()
    if opt.meteor_reward_weight > 0:
        #print('vid:', vid)
        _, meteor_score = Meteor_scorer.compute_score(gts, res__)
        #print('Meteor score:', _)
    else:
        meteor_score = 0

    scores =opt.meteor_reward_weight * np.array(meteor_score)
    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], sent_len, 1)
    #print('time consuming:',time.time()-start)

    return rewards

def get_self_critical_reward2(greedy_res, vid_info, gen_result, vocab, opt):
    vid, sentences_batch= vid_info

    start = time.time()
    batch_size,sent_len = gen_result.size()  # batch_size = sample_size * seq_per_img

    # get greedy decoding baseline
    gen_result = utils.decode_sequence(vocab, gen_result)
    greedy_res = utils.decode_sequence(vocab, greedy_res)

    # pdb.set_trace()
    res = OrderedDict()

    #gen_result = gen_result.data.cpu().numpy()
    #greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        #res[i] = [array_to_str(gen_result[i])]
        res[i] = [remove_nonascii(gen_result[i])]
    for i in range(batch_size):
        #res[batch_size + i] = [array_to_str(greedy_res[i])]
        res[batch_size + i] = [remove_nonascii(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(sentences_batch)):
        gts[i] = [remove_nonascii(sentences_batch[i])]

    #res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]  # for cider
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i%batch_size] for i in range(2 * batch_size)}

    #pdb.set_trace()
    if opt.meteor_reward_weight > 0:
        #print('vid:', vid)
        _, meteor_score = Meteor_scorer.compute_score(gts, res__)
    else:
        meteor_score=0
    if opt.meteor_reward_weight <1:
        _, cider_score = Cider_scorer.compute_score(gts, res_)
        #print('Meteor score:', _)
    else:
        cider_score = 0

    scores =opt.meteor_reward_weight * np.array(meteor_score) + (1-opt.meteor_reward_weight) * np.array(cider_score)
    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], sent_len, 1)
    #print('time consuming:',time.time()-start)

    return rewards
