#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm_early_fusion_cross_enc_blending_sum_time
export CUDA_VISIBLE_DEVICES=0
python -m ptls.pl_train_module --config-dir "conf/" --config-name data_fusion_early_fusion_cross_enc_blending_sum \
'exp_name=data_fusion_early_fusion_mod_embs_concat_100_full_no_noize' "~ifilters=data_fusion" \
"+ifilters=data_fusion_full"  'trainer.accumulate_grad_batches=64' \
'trainer.limit_val_batches=8192' 'data_module.valid_batch_size=4'