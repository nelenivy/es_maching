export PYTHONPATH="~/chankaev/src/:~/chankaev/src/pytorch-lifestream:~/chankaev/src/pytorch-lifestream/ptls:$PYTHONPATH"
echo $PYTHONPATH
python -m ptls.pl_train_module --config-dir "conf/" --config-name MDB_head_time_transf
# python -m ptls.pl_train_module --config-dir "conf/" --config-name cikm2016_head_time_transf