#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm_early_fusion_cross_enc_blending_sum_time
export CUDA_VISIBLE_DEVICES=1
python -m ptls.pl_train_module --config-dir "conf/" --config-name MBD_early_fusion_cross_enc_blending_sum \
"sec_encoder_class=early_fusion_matching.M3ConcatTimeSeqEncoderContainerSameMod" \
'exp_name=MBD_early_fusion_vanilla_cross_enc_100_no_sample_no_amt' "~data=MBD_early" "+data=MBD_early_100" \
"~trx=MBD_no_time" "+trx=MBD_no_time_no_amt"