# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.autograd import Variable
import torch.optim as optim
import sys
import logging
import shutil

import opts
from dataloader import *
import eval_utils
# from OLD import eval_utils_diff
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward2
from CaptionGenerator import CaptionGenerator

# from misc.tensorboard_logging import tf_Logger
from tensorboardX import SummaryWriter
import models
import cPickle

print(sys.executable)


def get_training_list(opt, logger):
    if opt.training_mode == 'pre_tap+cotrain':
        flag_training_whats = ['tap'] * opt.tap_epochs + ['cg'] * opt.cg_epochs + ['tap_cg'] * opt.tapcg_epochs

    elif opt.training_mode == 'cotrain':
        assert (opt.tap_epochs == 0) & (opt.cg_epochs == 0)
        flag_training_whats = ['tap_cg'] * opt.tapcg_epochs

    elif opt.training_mode == 'pre_cg':
        assert (opt.tap_epochs == 0)
        flag_training_whats = ['cg'] * opt.cg_epochs

    elif opt.training_mode == 'pre_LP_cg':
        assert (opt.tap_epochs == 0)
        flag_training_whats = ['LP_cg'] * opt.cg_epochs

    elif opt.training_mode == 'gt_tap_cg':
        assert (opt.tap_epochs == 0)
        flag_training_whats = ['gt_tap_cg'] * opt.cg_epochs

    elif opt.training_mode == 'pre_tap':
        assert (opt.cg_epochs == 0)
        flag_training_whats = ['tap'] * opt.tap_epochs
    elif opt.training_mode == "alter":
        assert (opt.cg_epochs == 0) and (opt.tap_epochs == 0)
        flag_training_whats = ['gt_tap_cg', 'tap_cg'] * opt.tapcg_epochs

    elif opt.training_mode == "alter2":
        assert (opt.cg_epochs == 0) and (opt.tap_epochs == 0)
        flag_training_whats = (['gt_tap_cg'] * 500 + ['tap_cg'] * 500) * opt.tapcg_epochs * 10
    elif opt.training_mode == "alter3":
        assert (opt.cg_epochs == 0) and (opt.tap_epochs == 0)
        flag_training_whats = ['gt_tap_cg'] * 5 * 10009 + (['gt_tap_cg'] * 500 + ['tap_cg'] * 500) * opt.tapcg_epochs
    else:
        raise AssertionError, 'training_mode is incorrect'

    logger.info(
        'traing_mode: {}, initial_lr:{}, decay_begin:{}, decay_step:{},decay_rate:{}, start_rl:{}'.format(
            opt.training_mode, opt.lr, opt.learning_rate_decay_start, opt.learning_rate_decay_every,
            opt.learning_rate_decay_rate, opt.self_critical_after))
    return flag_training_whats


def print_opt(opt, allmodels, logger):
    logger.info('The hyper-parameter configuration:')
    for key, item in opt._get_kwargs():
        logger.info('| {} = {}'.format(key, item))
    logger.info('The model configuration:')
    for model in allmodels:
        logger.info(model)



def build_floder_and_create_logger(opt):
    if opt.start_from != None:
        print('start training from id:{}'.format(opt.start_from))
        save_folder = os.path.join('save', opt.start_from)
        assert os.path.exists(save_folder)
    else:
        save_folder = os.path.join('save', opt.id)
        if not os.path.exists(save_folder):
            print('Saving folder "{}" does not exist, creating folder...'.format(save_folder))
            os.mkdir(save_folder)
            os.mkdir(os.path.join(save_folder, 'pred_sent'))
        else:
            assert 1==0 ,'parameter id error, folder {} exists'.format(save_folder)
            opt.id = opt.id + '_%s' % int(time.time())
            save_folder = os.path.join('save', opt.id)
            os.mkdir(save_folder)
            os.mkdir(os.path.join(save_folder, 'pred_sent'))
            print('Saving folder exists, change to a new folder {}'.format(save_folder))
        shutil.copytree('./models', os.path.join(save_folder, 'models'))
        shutil.copytree('./misc', os.path.join(save_folder, 'misc'))
        shutil.copyfile('./dataloader.py',
                        os.path.join(save_folder, 'dataloader.py'))
        shutil.copyfile('./dataloader.py',
                        os.path.join(save_folder, 'dataloader.py'))
        shutil.copyfile('./train.py', os.path.join(save_folder, 'train.py'))
        shutil.copyfile('./eval_utils.py', os.path.join(save_folder, 'eval_utils.py'))

    logger = logging.getLogger('DVC_train')
    formatter_log = "[%(asctime)s] %(message)s"
    formatter_log2 = "[%(asctime)s] %(message)s"
    logging.basicConfig(
        format=formatter_log,
        datefmt='%d %H:%M')
    logger.setLevel(logging.DEBUG)
    hdlr = logging.FileHandler(os.path.join(save_folder, 'train.log'))
    hdlr.setLevel(logging.DEBUG)
    hdlr.setFormatter(logging.Formatter(formatter_log2))
    logger.addHandler(hdlr)

    # tf_writer = tf.summary.FileWriter(os.path.join(save_folder, 'tf_summary'))
    tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary_train'))
    return save_folder, logger, tf_writer


def train(opt):
    exclude_opt = ['training_mode', 'tap_epochs', 'cg_epochs', 'tapcg_epochs', 'lr', 'learning_rate_decay_start',
                   'learning_rate_decay_every', 'learning_rate_decay_rate', 'self_critical_after',
                   'save_checkpoint_every',
                   'id', "pretrain", "pretrain_path", "debug", "save_all_checkpoint", "min_epoch_when_save"]

    save_folder, logger, tf_writer = build_floder_and_create_logger(opt)
    saved_info = {'best': {}, 'last': {}, 'history': {}}
    is_continue = opt.start_from != None

    if is_continue:
        infos_path = os.path.join(save_folder, 'info.pkl')
        with open(infos_path) as f:
            logger.info('load info from {}'.format(infos_path))
            saved_info = cPickle.load(f)
            pre_opt = saved_info[opt.start_from_mode]['opt']
            if vars(opt).get("no_exclude_opt", False):
                exclude_opt = []
            for opt_name in vars(pre_opt).keys():
                if (not opt_name in exclude_opt):
                    vars(opt).update({opt_name: vars(pre_opt).get(opt_name)})
                if vars(pre_opt).get(opt_name) != vars(opt).get(opt_name):
                    print('change opt: {} from {} to {}'.format(opt_name, vars(pre_opt).get(opt_name),
                                                                vars(opt).get(opt_name)))

    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.CG_vocab_size = loader.vocab_size
    opt.CG_seq_length = loader.seq_length

    # init training option
    epoch = saved_info[opt.start_from_mode].get('epoch', 0)
    iteration = saved_info[opt.start_from_mode].get('iter', 0)
    best_val_score = saved_info[opt.start_from_mode].get('best_val_score', 0)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    loader.iterators = saved_info[opt.start_from_mode].get('iterators', loader.iterators)
    loader.split_ix = saved_info[opt.start_from_mode].get('split_ix', loader.split_ix)
    opt.current_lr = vars(opt).get('current_lr', opt.lr)
    opt.m_batch = vars(opt).get('m_batch', 1)

    # create a tap_model,fusion_model,cg_model

    tap_model = models.setup_tap(opt)
    lm_model = CaptionGenerator(opt)
    cg_model = lm_model

    if is_continue:
        if opt.start_from_mode == 'best':
            model_pth = torch.load(os.path.join(save_folder, 'model-best.pth'))
        elif opt.start_from_mode == 'last':
            model_pth = torch.load(os.path.join(save_folder, 'model_iter_{}.pth'.format(iteration)))
        assert model_pth['iteration'] == iteration
        logger.info('Loading pth from {}, iteration:{}'.format(save_folder, iteration))
        tap_model.load_state_dict(model_pth['tap_model'])
        cg_model.load_state_dict(model_pth['cg_model'])

    elif opt.pretrain:
        print('pretrain {} from {}'.format(opt.pretrain, opt.pretrain_path))
        model_pth = torch.load(opt.pretrain_path)
        if opt.pretrain == 'tap':
            tap_model.load_state_dict(model_pth['tap_model'])
        elif opt.pretrain == 'cg':
            cg_model.load_state_dict(model_pth['cg_model'])
        elif opt.pretrain == 'tap_cg':
            tap_model.load_state_dict(model_pth['tap_model'])
            cg_model.load_state_dict(model_pth['cg_model'])
        else:
            assert 1==0, 'opt.pretrain error'

    tap_model.cuda()
    tap_model.train()  # Assure in training mode

    tap_crit = utils.TAPModelCriterion()

    tap_optimizer = optim.Adam(tap_model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    cg_model.cuda()
    cg_model.train()
    cg_optimizer = optim.Adam(cg_model.parameters(), lr=opt.lr,
                              weight_decay=opt.weight_decay)
    cg_crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    cg_optimizer = optim.Adam(cg_model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    allmodels = [tap_model, cg_model]
    optimizers = [tap_optimizer, cg_optimizer]

    if is_continue:
        tap_optimizer.load_state_dict(model_pth['tap_optimizer'])
        cg_optimizer.load_state_dict(model_pth['cg_optimizer'])

    update_lr_flag = True
    loss_sum = np.zeros(5)
    bad_video_num = 0
    best_epoch = epoch
    start = time.time()

    print_opt(opt, allmodels, logger)
    logger.info('\nStart training')

    # set a var to indicate what to train in current iteration: "tap", "cg" or "tap_cg"
    flag_training_whats = get_training_list(opt, logger)

    # Iteration begin
    while True:
        if update_lr_flag:
            if (epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0):
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.lr * decay_factor
            else:
                opt.current_lr = opt.lr
            for optimizer in optimizers:
                utils.set_lr(optimizer, opt.current_lr)
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(None)
            else:
                sc_flag = False
            update_lr_flag = False

        flag_training_what = flag_training_whats[epoch]
        if opt.training_mode == "alter2":
            flag_training_what = flag_training_whats[iteration]

        # get data
        data = loader.get_batch('train')

        if opt.debug:
            print('vid:', data['vid'])
            print('info:', data['infos'])

        torch.cuda.synchronize()

        if (data["proposal_num"] <= 0) or (data['fc_feats'].shape[0] <= 1):
            bad_video_num += 1  # print('vid:{} has no good proposal.'.format(data['vid']))
            continue

        ind_select_list, soi_select_list, cg_select_list, sampled_ids, = data['ind_select_list'], data[
            'soi_select_list'], data['cg_select_list'], data['sampled_ids']

        if flag_training_what == 'cg' or flag_training_what == 'gt_tap_cg':
            ind_select_list = data['gts_ind_select_list']
            soi_select_list = data['gts_soi_select_list']
            cg_select_list = data['gts_cg_select_list']

        tmp = [data['fc_feats'], data['att_feats'], data['lda_feats'], data['tap_labels'],
               data['tap_masks_for_loss'], data['cg_labels'][cg_select_list], data['cg_masks'][cg_select_list],
               data['w1']]

        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]

        c3d_feats, att_feats, lda_feats, tap_labels, tap_masks_for_loss, cg_labels, cg_masks, w1 = tmp

        if (iteration - 1) % opt.m_batch == 0:
            tap_optimizer.zero_grad()
            cg_optimizer.zero_grad()

        tap_feats, pred_proposals = tap_model(c3d_feats)
        tap_loss = tap_crit(pred_proposals, tap_masks_for_loss, tap_labels, w1)

        loss_sum[0] = loss_sum[0] + tap_loss.item()

        # Backward Propagation
        if flag_training_what == 'tap':
            tap_loss.backward()
            utils.clip_gradient(tap_optimizer, opt.grad_clip)
            if iteration % opt.m_batch == 0:
                tap_optimizer.step()
        else:
            if not sc_flag:
                pred_captions = cg_model(tap_feats, c3d_feats, lda_feats, cg_labels, ind_select_list, soi_select_list,
                                         mode='train')
                cg_loss = cg_crit(pred_captions, cg_labels[:, 1:], cg_masks[:, 1:])

            else:
                gen_result, sample_logprobs, greedy_res = cg_model(tap_feats, c3d_feats, lda_feats, cg_labels,
                                                                   ind_select_list, soi_select_list, mode='train_rl')
                sentence_info = data['sentences_batch'] if (flag_training_what != 'cg'  and flag_training_what!='gt_tap_cg') else data['gts_sentences_batch']

                reward = get_self_critical_reward2(greedy_res, (data['vid'], sentence_info), gen_result, vocab=loader.get_vocab(), opt=opt)
                cg_loss = rl_crit(sample_logprobs, gen_result, torch.from_numpy(reward).float().cuda())

            loss_sum[1] = loss_sum[1] + cg_loss.item()

            if flag_training_what == 'cg' or flag_training_what == 'gt_tap_cg' or flag_training_what=='LP_cg':
                cg_loss.backward()

                utils.clip_gradient(cg_optimizer, opt.grad_clip)
                if iteration % opt.m_batch == 0:
                    cg_optimizer.step()
                if flag_training_what == 'gt_tap_cg':
                    utils.clip_gradient(tap_optimizer, opt.grad_clip)
                    if iteration % opt.m_batch == 0:
                        tap_optimizer.step()
            elif flag_training_what == 'tap_cg':
                total_loss = opt.lambda1 * tap_loss + opt.lambda2 * cg_loss
                total_loss.backward()
                utils.clip_gradient(tap_optimizer, opt.grad_clip)
                utils.clip_gradient(cg_optimizer, opt.grad_clip)
                if iteration % opt.m_batch == 0:
                    tap_optimizer.step()
                    cg_optimizer.step()

                loss_sum[2] = loss_sum[2] + total_loss.item()

        torch.cuda.synchronize()

        # Updating epoch num
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Print losses, Add to summary
        if iteration % opt.losses_log_every == 0:
            end = time.time()
            losses = np.round(loss_sum / opt.losses_log_every, 3)
            logger.info(
                "iter {} (epoch {}, lr {}), avg_iter_loss({}) = {}, time/batch = {:.3f}, bad_vid = {:.3f}" \
                    .format(iteration, epoch, opt.current_lr, flag_training_what, losses,
                            (end - start) / opt.losses_log_every,
                            bad_video_num))

            tf_writer.add_scalar('lr', opt.current_lr, iteration)
            tf_writer.add_scalar('train_tap_loss', losses[0], iteration)
            tf_writer.add_scalar('train_tap_prop_loss', losses[3], iteration)
            tf_writer.add_scalar('train_tap_bound_loss', losses[4], iteration)
            tf_writer.add_scalar('train_cg_loss', losses[1], iteration)
            tf_writer.add_scalar('train_total_loss', losses[2], iteration)
            if sc_flag and (not flag_training_what=='tap'):
                tf_writer.add_scalar('avg_reward', np.mean(reward[:, 0]), iteration)
            loss_history[iteration] = losses
            lr_history[iteration] = opt.current_lr
            loss_sum = np.zeros(5)
            start = time.time()
            bad_video_num = 0

        # Evaluation, and save model
        if (iteration % opt.save_checkpoint_every == 0) and (epoch >= opt.min_epoch_when_save):
            eval_kwargs = {'split': 'val',
                           'val_all_metrics': 0,
                           'topN': 100,
                           }

            eval_kwargs.update(vars(opt))

            # eval_kwargs['num_vids_eval'] = int(491)
            eval_kwargs['topN'] = 100

            eval_kwargs2 = {'split': 'val',
                            'val_all_metrics': 1,
                            'num_vids_eval': 4917,
                            }
            eval_kwargs2.update(vars(opt))

            if not opt.num_vids_eval:
                eval_kwargs['num_vids_eval'] = int(4917.)
                eval_kwargs2['num_vids_eval'] = 4917

            crits = [tap_crit, cg_crit]
            pred_json_path_T = os.path.join(save_folder, 'pred_sent',
                                          'pred_num{}_iter{}.json')

            # if 'alter' in opt.training_mode:
            if flag_training_what == 'tap':
                eval_kwargs['topN'] = 1000
                predictions, eval_score, val_loss = eval_utils.eval_split(allmodels, crits, loader, pred_json_path_T.format(eval_kwargs['num_vids_eval'], iteration),
                                                                          eval_kwargs,
                                                                          flag_eval_what='tap')
            else:
                if vars(opt).get('fast_eval_cg', False) == False:
                    predictions, eval_score, val_loss = eval_utils.eval_split(allmodels, crits, loader, pred_json_path_T.format(eval_kwargs['num_vids_eval'], iteration),
                                                                              eval_kwargs,
                                                                              flag_eval_what='tap_cg')

                predictions2, eval_score2, val_loss2 = eval_utils.eval_split(allmodels, crits, loader, pred_json_path_T.format(eval_kwargs2['num_vids_eval'], iteration),
                                                                             eval_kwargs2,
                                                                             flag_eval_what='cg')

                if (not vars(opt).get('fast_eval_cg', False) == False)  or (not vars(opt).get('fast_eval_cg_top10', False) == False):
                        eval_score = eval_score2
                        val_loss = val_loss2
                        predictions = predictions2


            # else:
            #    predictions, eval_score, val_loss = eval_utils.eval_split(allmodels, crits, loader, pred_json_path,
            #                                                              eval_kwargs,
            #                                                              flag_eval_what=flag_training_what)

            f_f1 = lambda x, y: 2 * x * y / (x + y)
            f1 = f_f1(eval_score['Recall'], eval_score['Precision']).mean()
            if flag_training_what != 'tap':  # if only train tap, use the mean of precision and recall as final score
                current_score = np.array(eval_score['METEOR']).mean() * 100
            else:  # if train tap_cg, use avg_meteor as final score
                current_score = f1

            for model in allmodels:
                for name, param in model.named_parameters():
                    tf_writer.add_histogram(name, param.clone().cpu().data.numpy(), iteration, bins=10)
                    if param.grad is not None:
                        tf_writer.add_histogram(name + '_grad', param.grad.clone().cpu().data.numpy(), iteration,
                                                bins=10)

            tf_writer.add_scalar('val_tap_loss', val_loss[0], iteration)
            tf_writer.add_scalar('val_cg_loss', val_loss[1], iteration)
            tf_writer.add_scalar('val_tap_prop_loss', val_loss[3], iteration)
            tf_writer.add_scalar('val_tap_bound_loss', val_loss[4], iteration)
            tf_writer.add_scalar('val_total_loss', val_loss[2], iteration)
            tf_writer.add_scalar('val_score', current_score, iteration)
            if flag_training_what != 'tap':
                tf_writer.add_scalar('val_score_gt_METEOR', np.array(eval_score2['METEOR']).mean(), iteration)
                tf_writer.add_scalar('val_score_gt_Bleu_4', np.array(eval_score2['Bleu_4']).mean(), iteration)
                tf_writer.add_scalar('val_score_gt_CIDEr', np.array(eval_score2['CIDEr']).mean(), iteration)
            tf_writer.add_scalar('val_recall', eval_score['Recall'].mean(), iteration)
            tf_writer.add_scalar('val_precision', eval_score['Precision'].mean(), iteration)
            tf_writer.add_scalar('f1', f1, iteration)

            val_result_history[iteration] = {'val_loss': val_loss, 'eval_score': eval_score}

            if flag_training_what == 'tap':
                logger.info(
                    'Validation the result of iter {}, score(f1/meteor):{},\n all:{}'.format(iteration, current_score,
                                                                                             eval_score))
            else:
                mean_score = {k: np.array(v).mean() for k, v in eval_score.items()}
                gt_mean_score = {k: np.array(v).mean() for k, v in eval_score2.items()}

                metrics = ['Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L']
                gt_avg_score = np.array([v for metric, v in gt_mean_score.items() if metric in metrics]).sum()
                logger.info(
                    'Validation the result of iter {}, score(f1/meteor):{},\n all:{}\n mean:{} \n\n gt:{} \n mean:{}\n avg_score: {}'.format(
                        iteration, current_score,
                        eval_score, mean_score, eval_score2, gt_mean_score, gt_avg_score))

            # Save model .pth
            saved_pth = {'iteration': iteration,
                         'cg_model': cg_model.state_dict(),
                         'tap_model': tap_model.state_dict(),
                         'cg_optimizer': cg_optimizer.state_dict(),
                         'tap_optimizer': tap_optimizer.state_dict(),
                         }

            if opt.save_all_checkpoint:
                checkpoint_path = os.path.join(save_folder, 'model_iter_{}.pth'.format(iteration))
            else:
                checkpoint_path = os.path.join(save_folder, 'model.pth')
            torch.save(saved_pth, checkpoint_path)
            logger.info('Save model at iter {} to checkpoint file {}.'.format(iteration, checkpoint_path))

            # save info.pkl
            if current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                saved_info['best'] = {'opt': opt,
                                      'iter': iteration,
                                      'epoch': epoch,
                                      'iterators': loader.iterators,
                                      'flag_training_what': flag_training_what,
                                      'split_ix': loader.split_ix,
                                      'best_val_score': best_val_score,
                                      'vocab': loader.get_vocab(),
                                      }

                best_checkpoint_path = os.path.join(save_folder, 'model-best.pth')
                torch.save(saved_pth, best_checkpoint_path)
                logger.info('Save Best-model at iter {} to checkpoint file.'.format(iteration))

            saved_info['last'] = {'opt': opt,
                                  'iter': iteration,
                                  'epoch': epoch,
                                  'iterators': loader.iterators,
                                  'flag_training_what': flag_training_what,
                                  'split_ix': loader.split_ix,
                                  'best_val_score': best_val_score,
                                  'vocab': loader.get_vocab(),
                                  }
            saved_info['history'] = {'val_result_history': val_result_history,
                                     'loss_history': loss_history,
                                     'lr_history': lr_history,
                                     }
            with open(os.path.join(save_folder, 'info.pkl'), 'w') as f:
                cPickle.dump(saved_info, f)
                logger.info('Save info to info.pkl')

            # Stop criterion
            if epoch >= len(flag_training_whats):
                tf_writer.close()
                break


if __name__ == '__main__':
    opt = opts.parse_opts()
    opt.fc_feat_size = opt.hidden_dim
    train(opt)
    # myDebug(opt)
