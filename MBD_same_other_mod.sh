#export PYTHONPATH="~/chankaev/src/:~/chankaev/src/pytorch-lifestream:~/chankaev/src/pytorch-lifestream/ptls:$PYTHONPATH"
echo $PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#python -m train_by_optuna --config-dir "conf/" --config-name optuna
python -m ptls.pl_train_module --config-dir "conf/" --config-name MBD_head_time_transf_same \
"pl_module.loss.only_other_mod=True" 'exp_name=MBD_late_fusion_other_mod_full'
# python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_transf