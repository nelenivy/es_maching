#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm_early_fusion_cross_enc_blending_sum_time
export CUDA_VISIBLE_DEVICES=0
python -m ptls.pl_train_module --config-dir "conf/" --config-name data_fusion_early_fusion_cross_enc_blending_sum \
'exp_name=data_fusion_early_fusion_mod_embs_concat_100_post_norm_no_amt_dyn_pos_3' \
'~seq_encoder=early_fusion_time' "+seq_encoder=early_fusion_time_dyn"
