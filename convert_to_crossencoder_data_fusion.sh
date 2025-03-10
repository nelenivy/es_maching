
#--checkpoint "/home/amshestov/smchankaev/matching_src/lightning_logs/data_fusion_late_fusion/version_18/checkpoints/epoch=24-step=1275.ckpt"
# Define the number of folds
#num_folds=5  # Adjust this based on how many folds you have
python -m m3_crossencoder_convert \
        --parquet_in "/home/amshestov/smchankaev/MBD/scenario_datafusion/data/mm_dataset_fold/fold=0" \
        --parquet_in "/home/amshestov/smchankaev/MBD/scenario_datafusion/data/mm_dataset_fold/fold=1" \
        --parquet_in "/home/amshestov/smchankaev/MBD/scenario_datafusion/data/mm_dataset_fold/fold=2" \
        --parquet_in "/home/amshestov/smchankaev/MBD/scenario_datafusion/data/mm_dataset_fold/fold=3" \
        --yaml_in "/home/amshestov/smchankaev/matching_src/conf/data_fusion_late_fusion.yaml" \
        --checkpoint "/home/amshestov/smchankaev/matching_src/lightning_logs/data_fusion_late_fusion_other_mod_200/version_0/checkpoints/epoch=49-step=2550.ckpt" \
        --parquet_out "/home/amshestov/smchankaev/data/data_fusion_cross_enc_100/train" \
        --col_id user_id --topn 100 \
        --mod1_cols mcc_code --mod1_cols currency_rk --mod1_cols transaction_amt --mod1_name trx \
        --mod2_cols cat_id --mod2_cols level_0 --mod2_cols level_1 --mod2_cols level_2 --mod2_name click 

python -m m3_crossencoder_convert \
        --parquet_in "/home/amshestov/smchankaev/MBD/scenario_datafusion/data/mm_dataset_fold/fold=4" \
        --yaml_in "/home/amshestov/smchankaev/matching_src/conf/data_fusion_late_fusion.yaml" \
        --checkpoint "/home/amshestov/smchankaev/matching_src/lightning_logs/data_fusion_late_fusion_other_mod_200/version_0/checkpoints/epoch=49-step=2550.ckpt" \
        --parquet_out "/home/amshestov/smchankaev/data/data_fusion_cross_enc_100/val" \
        --col_id user_id --topn 100 \
        --mod1_cols mcc_code --mod1_cols currency_rk --mod1_cols transaction_amt --mod1_name trx \
        --mod2_cols cat_id --mod2_cols level_0 --mod2_cols level_1 --mod2_cols level_2 --mod2_name click 
# Iterate over each fold
#for ((fold=-1; fold<num_folds; fold++)); do
#    echo "Processing fold=$fold"
    
#    python -m m3_crossencoder_convert \
#        --parquet_in "/home/amshestov/smchankaev/data/mm_dataset/fold=$fold" \
#        --yaml_in "/home/amshestov/smchankaev/matching_src/conf/MBD_head_time_transf_same.yaml" \
#        --checkpoint "/home/amshestov/smchankaev/matching_src/lightning_logs/MBD_late_fusion_other_mod/version_0/checkpoints/epoch=14-step=18345.ckpt" \
#        --parquet_out "/home/amshestov/smchankaev/data/mm_cross_enc/fold=$fold" \
#        --col_id client_id
    
#done
