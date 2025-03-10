#export PYTHONPATH="~/chankaev/src/:~/chankaev/src/pytorch-lifestream:~/chankaev/src/pytorch-lifestream/ptls:$PYTHONPATH"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#python -m train_by_optuna --config-dir "conf/" --config-name optuna
python -m ptls.pl_train_module --config-dir "conf/" --config-name MBD_head_time_transf_same \
"data_module.train_data.splitter._target_=ptls.frames.coles.split_strategy.NoSplit" \
"~data_module.train_data.splitter.split_count" \
"~data_module.train_data.splitter.cnt_min" \
"~data_module.train_data.splitter.cnt_max" \
"data_module.train_batch_size=256" \
'exp_name=MBD_late_fusion_no_split_256'
# python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_transf