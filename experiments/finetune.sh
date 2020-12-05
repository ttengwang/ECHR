gpuid=0
SST_Name=SST
EC_Name=EC_C3D
JT_Name=SST_EC_C3D

CUDA_VISIBLE_DEVICES=${gpuid} python train_baseline_continue_rl_CSVTversion_more_feature.py  --training_mode 'tap_cg' --tap_epoch 0 --cg_epoch 0 --tapcg_epoch 10  --caption_model 'three_stream' --CG_num_layers 3  --other_feature lda --input_lda_path 'data/lda_data1205/feats_100_doc_feat1_train_val_test_Tcov_reformat.h5' --lda_dim 100 --id ${JT_Name} --CG_input_feats_type '' --CG_init_feats_type '' --video_context_type 'VL' --event_context_type 'ER3' --clip_context_type 'CC' --lr 1e-6 --learning_rate_decay_start 8  --learning_rate_decay_every 3  --learning_rate_decay_rate 0.5 --min_epoch_when_save 8 --save_all --fast_eval_cg --use_bottomup_feature 1 --video_dim 1024 --reduce_input_dim_layer 1 --raw_input_dim 10240 --K 256 --w1_json 'data/w1_256_c3d32_iou0.5.json' --pretrain tap_cg --pretrain_path save/${EC_Name}/model-best.pth--losses_log_every 2000 --save_checkpoint_every 10000