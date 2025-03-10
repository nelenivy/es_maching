export CUDA_VISIBLE_DEVICES=0
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm_early_fusion_cross_enc_blending_sum_time
python -m ptls.pl_train_module --config-dir "conf/" --config-name MBD_early_fusion_cross_enc_blending_sum \
"~data=MBD_early" "+data=MBD_early_100" \
'exp_name=MBD_early_fusion_blending_100_no_sample_no_amt_concat_dyn_3_segm' \
"~trx=MBD_no_time" "+trx=MBD_no_time_no_amt" '~seq_encoder=early_fusion' "+seq_encoder=early_fusion_dyn_segm"