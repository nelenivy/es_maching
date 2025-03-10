#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm_early_fusion_cross_enc_blending_sum_time
export CUDA_VISIBLE_DEVICES=1
python -m ptls.pl_train_module --config-dir "conf/" --config-name data_fusion_early_fusion_cross_enc_blending_sum \
'exp_name=data_fusion_early_fusion_vanilla_cross_enc_100_full' "~ifilters=data_fusion" \
"+ifilters=data_fusion_full"  'trainer.accumulate_grad_batches=64' \
'trainer.limit_val_batches=8192' 'data_module.valid_batch_size=4' \
"sec_encoder_class=early_fusion_matching.M3ConcatTimeSeqEncoderContainerSameMod"