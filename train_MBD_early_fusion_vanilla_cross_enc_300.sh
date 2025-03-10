#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm_early_fusion_cross_enc_blending_sum_time
export CUDA_VISIBLE_DEVICES=1
python -m ptls.pl_train_module --config-dir "conf/" --config-name MBD_early_fusion_cross_enc_blending_sum \
"sec_encoder_class=early_fusion_matching.M3ConcatTimeSeqEncoderContainerSameMod" \
'exp_name=MBD_early_fusion_vanilla_cross_enc_300_no_sample' "~data=MBD_early" "+data=MBD_early_300"