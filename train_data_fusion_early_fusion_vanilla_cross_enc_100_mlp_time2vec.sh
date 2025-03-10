#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm_early_fusion_cross_enc_blending_sum_time
export CUDA_VISIBLE_DEVICES=1
python -m ptls.pl_train_module --config-dir "conf/" --config-name data_fusion_early_fusion_cross_enc_blending_sum \
"sec_encoder_class=early_fusion_matching.M3ConcatTimeSeqEncoderContainerSameMod" \
'exp_name=data_fusion_early_fusion_vanilla_cross_enc_100_post_norm_no_amt_time2vec_rel' \
'~seq_encoder=early_fusion_time' "+seq_encoder=early_fusion_time_time2vec_rel"
