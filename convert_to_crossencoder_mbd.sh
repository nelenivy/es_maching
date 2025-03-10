
# Define the number of folds
#num_folds=5  # Adjust this based on how many folds you have
python -m m3_crossencoder_convert \
        --parquet_in "/home/amshestov/smchankaev/data/mm_dataset/train/fold=0" \
        --parquet_in "/home/amshestov/smchankaev/data/mm_dataset/train/fold=1" \
        --parquet_in "/home/amshestov/smchankaev/data/mm_dataset/train/fold=2" \
        --parquet_in "/home/amshestov/smchankaev/data/mm_dataset/train/fold=3" \
        --parquet_in "/home/amshestov/smchankaev/data/mm_dataset/train/fold=4" \
        --yaml_in "/home/amshestov/smchankaev/matching_src/conf/MBD_head_time_transf_same.yaml" \
        --checkpoint "/home/amshestov/smchankaev/matching_src/lightning_logs/MBD_late_fusion_other_mod_full/version_5/checkpoints/epoch=13-step=17122.ckpt" \
        --parquet_out "/home/amshestov/smchankaev/data/mm_cross_enc_300/train" \
        --col_id client_id --topn 300

python -m m3_crossencoder_convert \
        --parquet_in "/home/amshestov/smchankaev/data/mm_dataset/val/fold=-1" \
        --yaml_in "/home/amshestov/smchankaev/matching_src/conf/MBD_head_time_transf_same.yaml" \
        --checkpoint "/home/amshestov/smchankaev/matching_src/lightning_logs/MBD_late_fusion_other_mod_full/version_5/checkpoints/epoch=13-step=17122.ckpt" \
        --parquet_out "/home/amshestov/smchankaev/data/mm_cross_enc_300/val" \
        --col_id client_id --topn 300
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
