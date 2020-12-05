from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models
import argparse
import torch
import logging

from CaptionGenerator import CaptionGenerator
from dataloader import *

def create_logger(save_folder):
    logger = logging.getLogger('DVC_val')
    formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s"
    logging.basicConfig(
        format=formatter_log,
        datefmt='%d %b %H:%M:%S')
    logger.setLevel(logging.DEBUG)
    hdlr = logging.FileHandler(os.path.join(save_folder, 'val.log'))
    hdlr.setLevel(logging.DEBUG)
    hdlr.setFormatter(logging.Formatter(formatter_log))
    logger.addHandler(hdlr)
    return logger

def main(val_opt, logger):

    # Load infos
    folder_path = os.path.join('save', val_opt.folder_id)
    infos_path = os.path.join(folder_path, 'info.pkl')

    with open(infos_path) as f:
        logger.info('load info from {}'.format(infos_path))
        opt = cPickle.load(f)['best']['opt']
        vars(opt).update(vars(val_opt))

    # Create the Data Loader instance
    if opt.flag_eval_what == "cg_extend":
        opt.prop_sample_num = 100

    loader = DataLoader(opt)

    opt.CG_vocab_size = loader.vocab_size
    opt.CG_seq_length = loader.seq_length

    tap_model = models.setup_tap(opt)
    cg_model = CaptionGenerator(opt)

    if opt.model_path:
        model_path = opt.model_path
    else:
        model_path = os.path.join(folder_path, 'model-best.pth')
    while not os.path.exists(model_path):
        print('model.pth does not exists, waiting...')
        time.sleep(300)

    logger.debug('Loading model from {}'.format(model_path))

    loaded_pth = torch.load(model_path)
    pth_iter = loaded_pth['iteration']
    tap_model.load_state_dict(loaded_pth['tap_model'])
    tap_model.cuda()
    tap_model.eval()

    cg_model.load_state_dict(loaded_pth['cg_model'])
    cg_model.cuda()
    cg_model.eval()

    if opt.no_language_eval:
        pred_json_path = os.path.join('save', opt.folder_id, 'best.json')
    else:
        pred_json_path = os.path.join('save', opt.folder_id, 'pred_json_iter{}_eval{}_num_vids{}_topN{}_SCORETHRES{}_NMSTHRES{}_rerank{}_BEAM_SIZE{}.json'.format(
                                                                           pth_iter,opt.flag_eval_what, opt.num_vids_eval, opt.topN, opt.val_score_thres, opt.nms_threshold, int(opt.reranking),
                                                                           opt.beam_size))
    allmodels = [tap_model, cg_model]
    eval_kwargs = {'split': 'val',
                   'language_eval':(not opt.no_language_eval),
                   'get_eval_loss': 0
                   }
    eval_kwargs.update(vars(opt))
    eval_kwargs['language_eval'] = (not opt.no_language_eval)

    if eval_kwargs['tap_model']=='sst_1stage':
        import eval_utils_1stage_new as eval_utils
    else:
        import eval_utils
    predictions, eval_score, _ = eval_utils.eval_split(allmodels, [None, None], loader, pred_json_path,
                                                       eval_kwargs, flag_eval_what=opt.flag_eval_what)
    avg_eval_score = {key: np.array(value).mean() for key, value in eval_score.items()}
    logger.info(
        'Validation the result iter_{} num_vids{} topN {} SCORE_THRES{} NMS_THRES{} rerank{} BEAM_SIZE{}\n: {}\n avg_score:\n{}'.format(
            pth_iter, opt.num_vids_eval, opt.topN, opt.val_score_thres, opt.nms_threshold, opt.reranking, opt.beam_size, eval_score,
            avg_eval_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--val_all_metrics', type=int, default=1,
                        help='1 for val all metrcis(BLEU, METEOR, ROUGLE, CIDER), 0 for only val METEOR')

    parser.add_argument('--flag_eval_what', type=str, default='tap_cg',
                        help='tap,lm,or tap_lm, SOTA_TEP')

    parser.add_argument('--dataset', type=str, default='ActivityNet',
                        help='Name of the data class to use from data.py')

    parser.add_argument('--folder_id', type=str, default='sst_h')

    parser.add_argument('--model_path', type=str, default='')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='if > 0 then overrule, otherwise load from checkpoint.')

    #parser.add_argument('--language_eval', type=int, default=1,
    #                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')

    parser.add_argument('--sample_max', type=int, default=1,
                        help='1 = sample argmax words. 0 = sample from distributions.')

    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

    parser.add_argument('--split', type=str, default='val',
                        help='if running on MSCOCO images, which split to use: val|test|train')
    parser.add_argument('--debug', type=bool, default=False, )

    parser.add_argument('--num_vids_eval', type=int, default=491)

    parser.add_argument('--val_score_thres', type=float, default=0,
                        help='parameter in def get1000()')

    parser.add_argument('--nms_threshold', type=float, default=0)

    parser.add_argument('--topN', type=int, default=1000)

    parser.add_argument('--reranking', type=int, default=0)

    parser.add_argument("--old_loader", action='store_true')

    parser.add_argument("--no_language_eval", action='store_true')

    parser.add_argument('--SOTA_json', type=str, default='data/SOTA_TEP/Top10.json')


    opt = parser.parse_args()

    logger = create_logger(os.path.join('save', opt.folder_id))

    print(vars(opt))
    main(opt, logger)
