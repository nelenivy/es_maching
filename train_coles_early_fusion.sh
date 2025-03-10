export PYTHONPATH="~/shestov/src/matching/:~/shestov/src/:~/shestov/src/pytorch-lifestream:~/shestov/src/pytorch-lifestream/ptls:$PYTHONPATH"
echo $PYTHONPATH
python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm_early_fusion_cross_enc