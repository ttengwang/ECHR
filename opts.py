import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    # ID of this run

    parser.add_argument('--id', type=str, default='default',
                        help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--comment', type=str,
                        help='please write down your comment about this run!!')

    parser.add_argument('--debug', action='store_true')

    # INPUT DATA PATH

    parser.add_argument('--dataset', type=str, default='ActivityNet',
                        help='Name of the data class to use from data.py')

    parser.add_argument('--video_json', type=str,
                        default='data/video_data_with_annotation.json',
                        help='location of the dataset')

    # parser.add_argument('--input_c3d_dir', type=str,
    #                    default='data/sub_activitynet_v1-3.c3d_sampling_per64frame.hdf5',
    #                    help='path to the directory containing the preprocessed fc feats')

    parser.add_argument('--input_c3d_dir2', type=str,
                        default='data/c3d_npy',  # 'data/c3d_npy'
                        help='path to the directory containing the preprocessed fc feats')

    parser.add_argument('--input_lda_path', type=str,
                        default='data/lda_data1205/feats_100_doc_feat1_train_val_test_Tcov_reformat.h5')
    # data/lda_data2/feats_200_doc_feat1_train_val_test_Tcov_reformat.h5
    # 'data/lda_data1205/feats_64_doc_scorefeat1_train_val_test_reformat.h5')  # using doc_feature

    parser.add_argument('--video_data_for_cg', type=str, default='data/train_val_video_data_withID_6.0.json',
                        help='path to the json file containing additional info and vocab')

    parser.add_argument('--train_label_for_cg', type=str, default='data/train_label_for_lm_6.0.hdf5',
                        help='path to the h5file containing the preprocessed train dataset')

    parser.add_argument('--val_label_for_cg', type=str, default='data/val_label_for_lm_6.0.hdf5',
                        help='path to the h5file containing the preprocessed val dataset')

    parser.add_argument('--w1_json', type=str, default='data/w1_256_c3d64_iou0.5.json')

    parser.add_argument('--start_from', type=str, default=None)

    parser.add_argument('--start_from_mode', type=str, default="last")

    parser.add_argument('--no_exclude_opt', action='store_true')

    parser.add_argument('--pretrain', type=str, default='tap', help='tap,cg,tap_cg')

    parser.add_argument('--pretrain_path', type=str, default='save/1102_TAP32IOU0.5k128/model_iter_34000.pth',
                        help='path of .pth')

    parser.add_argument('--use_2stream_feature', type=int, default=0)

    parser.add_argument('--use_c3d_feature', type=int, default=1)

    parser.add_argument('--use_bottomup_feature', type=int, default=0)

    parser.add_argument('--input_twostream_dir', type=str, default='data/activitynet_feature_cuhk/full_feature')
    # MODEL OPTION

    # TAP MODEL
    parser.add_argument('--tap_model', type=str, default="SST",
                        help='SST, DAPs, Diff_SST')

    parser.add_argument('--tap_rnn_type', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')

    parser.add_argument('--rnn_num_layers', type=int, default=2,
                        help='Number of layers in rnn')

    parser.add_argument('--rnn_dropout', type=float, default=0.5,
                        help='dropout used in rnn')

    parser.add_argument('--video_dim', type=int, default=500,
                        help='dimensions of video (C3D) features')
    parser.add_argument('--raw_input_dim', type=int, default=10240)

    parser.add_argument('--reduce_input_dim_layer', type=int, default=0)

    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='')

    parser.add_argument('--K', type=int, default=256,
                        help='Number of proposals')

    parser.add_argument('--prop_sample_num', type=int, default=64,
                        help='Number of proposals')

    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='threshold above which we say something is positive')

    parser.add_argument('--iou_threshold_for_good_proposal', type=float, default=0.8,
                        help='threshold above which proposals can be input in cg_model')

    # SEMANTIC FEATURES
    parser.add_argument('--other_features', type=str, nargs='+', default=['lda'])  # ['lda']

    parser.add_argument('--lda_dim', type=int, default=200,
                        help='dimensions of lda scene features')

    # FUSION MODEL
    parser.add_argument('--fusion_model', type=str, default='TSRM8',
                        help='temp_relation, MA, simple_e_h, MA_twostream')

    parser.add_argument('--use_posit', type=int, default=1,
                        help='if use_posit in MA_attention, only available when using MA_attention')

    parser.add_argument('--n_head', type=int, default=16, help='')
    parser.add_argument('--d_feats', type=int, default=512, help='')
    # parser.add_argument('--d_pos_emb', type=int, default='512', help='')
    parser.add_argument('--d_o', type=int, default=512, help='')
    parser.add_argument('--fST_type', type=str, default='fST0', help='')

    parser.add_argument('--CG_input_feats_type', type=str, default='', help='V+E+C')
    parser.add_argument('--CG_init_feats_type', type=str, default='', help='V_E_C')

    parser.add_argument('--video_context_type', type=str, default='VL+VC+VH')
    parser.add_argument('--video_context_dim', type=int, default=0)

    parser.add_argument('--event_context_type', type=str, default='EL+EC+EH+ER1+ER2+ER3')
    parser.add_argument('--event_context_dim', type=int, default=0)

    parser.add_argument('--clip_context_type', type=str, default='CC+CH')
    parser.add_argument('--clip_context_dim', type=int, default=0)

    # LDA MODEL
    # parser.add_argument('--lda_hidden_size', type=int, default=2048)
    # parser.add_argument('--lda_input_size', type=int, default=500)
    # parser.add_argument('--lda_output_size', type=int, default=200)

    # cg MODEL
    parser.add_argument('--caption_model', type=str, default="show_attend_tell",
                        help='shw_attend_tell, allimg')

    parser.add_argument('--CG_rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')

    parser.add_argument('--CG_num_layers', type=int, default=1,
                        help='number of layers in the RNN')

    parser.add_argument('--CG_rnn_type', type=str, default='lstm',
                        help='rnn, gru, or lstm')

    parser.add_argument('--CG_input_encoding_size', type=int, default=512,
                        help='the encoding size of each token in the vocabulary, and the image.')

    parser.add_argument('--CG_att_hid_size', type=int, default=512,
                        help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')

    parser.add_argument('--CG_fc_feat_size', type=int, default=512,
                        help='2048 for resnet, 4096 for vgg')

    parser.add_argument('--CG_drop_prob', type=float, default=0.5,
                        help='strength of dropout in the Language Model RNN')

    # DATA LOADER AND PREPROCESSING

    parser.add_argument('--shuffle', type=int, default=1,
                        help='whether to shuffle the data')

    parser.add_argument('--nthreads', type=int,
                        default=4)

    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of training samples to train with')

    parser.add_argument("--dropsent_mode", type=str, default="nodrop", help="nodrop, insert or truncate")

    # OPTIMIZER FOR TRAINING

    parser.add_argument('--training_mode', type=str, default='pre_tap+cotrain',
                        help='cotrain, pre_cg, pre_tap, pre_tap+cotrain')

    parser.add_argument('--tap_epochs', type=int, default=3)

    parser.add_argument('--cg_epochs', type=int, default=0)

    parser.add_argument('--tapcg_epochs', type=int, default=20, help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--m_batch', type=int, default=1)

    parser.add_argument('--lr', type=float, default=5e-5,
                        help='initial learning rate')

    parser.add_argument('--lambda1', type=float, default=0.01, help='tap_loss trade-off')

    parser.add_argument('--lambda2', type=float, default=1, )

    # parser.add_argument('--lambda3', type=float, default=1,)

    parser.add_argument('--grad_clip', type=float, default=100.,  # 5.,
                        help='clip gradients at this value')

    parser.add_argument('--optim', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')

    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for adam')

    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')

    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                        help='at what iteration to start decay gt probability')

    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                        help='every how many iterations thereafter to gt probability')

    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')

    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    parser.add_argument('--learning_rate_decay_start', type=float, default=8)

    parser.add_argument('--learning_rate_decay_every', type=float, default=3)

    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5)

    parser.add_argument('--self_critical_after', type=int, default=135)

    parser.add_argument('--meteor_reward_weight', type=float, default=1)

    parser.add_argument('--reverse_w0', action='store_true')

    # SAVE AND LOG

    parser.add_argument('--losses_log_every', type=int, default=2000,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

    parser.add_argument('--min_epoch_when_save', type=int, default=-1)

    parser.add_argument('--save_checkpoint_every', type=int, default=10000,
                        help='how often to save a model checkpoint (in iterations)?')

    parser.add_argument('--save_all_checkpoint', action='store_true')

    parser.add_argument('--checkpoint_path', type=str, default='save',
                        help='directory to store checkpointed models')

    # Evaluate options

    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')

    parser.add_argument('--num_vids_eval', type=int, default=0,
                        help='Number of videos to evaluate at each pass')

    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    parser.add_argument('--fast_eval_cg', action='store_true')
    parser.add_argument('--fast_eval_for_challenge', action='store_true')

    # For 1stageATT
    parser.add_argument('--lambda3', type=float, default=1., )

    parser.add_argument('--crit_type', type=str, default='mse')  # 'mse, ce'
    parser.add_argument('--diff', type=int, default=0, )
    parser.add_argument('--data_type', type=str, default='rescale')  # 'rescale, dot'

    parser.add_argument('--SOTA_json', type=str, default='data/SOTA_TEP/sst_top100_evalset.json')  # 'rescale, dot'

    args = parser.parse_args()

    args.use_lda = True if (args.other_features and ('lda' in args.other_features)) else False
    print('use lda feature:{}'.format(args.use_lda))

    if 'L' in args.video_context_type:
        assert args.use_lda == True

    if args.debug:
        args.min_epoch_when_save = 0
        args.save_checkpoint_every = 100
        args.losses_log_every = 50
        args.num_vids_eval = 10
        args.shuffle = 0
    return args
