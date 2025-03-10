#export PYTHONPATH="~/shestov/src/matching/:~/shestov/src/:~/shestov/src/pytorch-lifestream:~/shestov/src/pytorch-lifestream/ptls:$PYTHONPATH"
echo $PYTHONPATH
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time
python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_transf
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_tlbn
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_no_split
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_resample
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_stack
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_elem_patch
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_elem_patch_tr
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_inverted
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_transf_dual
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_patch
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_patch_tr
#python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm_mid_fusion_time