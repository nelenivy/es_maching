import numpy as np
import pyarrow as pa
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from m3_dataset_dists import M3DatasetEmbeddingsCalc
from ptls.data_load.datasets.parquet_dataset import ParquetFiles
from collections import defaultdict
from matching_bank_x_rmb import first
from m3_utils import batch_nearest_neighbors
from math import ceil
import os

class M3PairsComposer(object):
    def __init__(self, embs, uids, neg_pairs_rate, topn):
        self.neg_pairs_rate = neg_pairs_rate
        self.embs = embs
        self.uids = uids
        self.N_NEIGHBOURS = topn
        self.neg_pair_sample_rate = float(self.neg_pairs_rate) / self.N_NEIGHBOURS
        random.seed(42)
        
    def nearest_neighbors(self):
        print("nearest_neighbors")
        embs_to_nn = [emb for k, emb in self.embs.items()]
        self.neighb_inds = batch_nearest_neighbors(embs_to_nn[0], embs_to_nn[1], K=self.N_NEIGHBOURS, batch_size=256)
        self.neighb_inds = {k: neighb_ind for (k, emb), neighb_ind in zip(self.embs.items(), self.neighb_inds)}
        
    def neg_pairs(self):
        print("neg_pairs")
        self.neg_pairs = defaultdict(list)
        already_seen = set()
        mod_names = list(self.embs.keys())
        print(self.uids)
        
        for mod_num, mod in enumerate(mod_names):
            other_mod_num = int(not(mod_num))
            
            for i in tqdm(range(self.uids.shape[0])):
                curr_elem = self.uids[i]
                for ind in range(self.N_NEIGHBOURS):
                    curr_neighb_ind = self.neighb_inds[mod][i, ind]      
                    curr_neighb = self.uids[curr_neighb_ind]
                    
                    if curr_neighb == curr_elem:
                        continue
                    
                    skip = (random.random() > self.neg_pair_sample_rate)
                    
                    if skip:
                        continue
                        
                    pair = ((curr_elem, curr_neighb)) if mod_num == 0 else ((curr_neighb, curr_elem))
                    str_pair = str(pair[0]) + " " + str(pair[1])
                    if not (str_pair in already_seen):
                        already_seen.add(str_pair)
                        for j in range(2):
                            self.neg_pairs[mod_names[j]].append(pair[j])
                            
        self.neg_pairs = {mod: np.array(pairs) for mod, pairs in self.neg_pairs.items()}
        print(self.neg_pairs)       
        
    def sample_negatives(self):
        print("sample_negatives")
        print(self.neg_pairs)
        #number of negatives can be greater than needed, so we sample and cut those which exeed
        neg_pairs_perm = np.random.permutation(first(self.neg_pairs.values()).shape[0])
        pos_pairs_num = self.uids.shape[0]
        neg_pairs_num = min(int(self.neg_pairs_rate * pos_pairs_num), len(neg_pairs_perm))
        neg_pairs_perm = neg_pairs_perm[:neg_pairs_num]
        self.neg_pairs = {mod_name: neg_pairs_mod[neg_pairs_perm] for mod_name, neg_pairs_mod in self.neg_pairs.items()}        
    
    def process(self):
        self.nearest_neighbors()
        self.neg_pairs()
        self.sample_negatives()
    
class CrossEncoderConverter:
    def __init__(self, parquet_in, yaml_in, checkpoint, neg_pairs_rate, parquet_out, 
                 col_id='epk_id', topn=30, add_mod_cols=None):
        if type(parquet_in) is str:
            parquet_in = [parquet_in]
         
        self.parquet_out = parquet_out
        self.parquet_in = parquet_in
        self.col_id = col_id
        
        self.embeddings_computer = M3DatasetEmbeddingsCalc(self.parquet_in, yaml_in, checkpoint, col_id=col_id)
        self.m3_embs, self.uids = self.embeddings_computer.process()
        self.mod_names = list(self.m3_embs.keys())
        self.pairs_composer = M3PairsComposer(self.m3_embs, self.uids, neg_pairs_rate, topn)
        self.topn = topn
        self.add_mod_cols = add_mod_cols
        
    def read_data(self):
        print("read_data")        
        self.data_in = []
        parquet_helper = ParquetFiles(self.parquet_in)
        
        for f_name in tqdm(parquet_helper.data_files):
            curr_data = pd.read_parquet(f_name)
            
            if self.col_id in curr_data.columns:
                curr_data = curr_data.set_index(self.col_id, drop=False)
            print(f_name, curr_data.shape)
            self.data_in.append(curr_data)
            
        self.data_in = pd.concat(self.data_in)
        print('finish data reading', self.data_in.shape)
        self.split_mod_data()
        
    def split_mod_data(self):
        splited_data = {}
        
        #print("modality_split", {k:v for k,v in feature_arrays.items()})
        for mod_name in self.mod_names:
            curr_mod_cols = [k for k in self.data_in.columns if k.startswith(mod_name)]
            curr_mod_cols += self.add_mod_cols.get(mod_name, [])
            splited_data[mod_name] = self.data_in[curr_mod_cols]
            
        self.data_in = splited_data
            
    def make_data(self):
        self.read_data()
        self.pairs_composer.process()
        neg_data = pd.concat(
            [self.data_in[mod].loc[mod_neg_col, :].reset_index(drop=True, inplace=False)
             for mod, mod_neg_col in self.pairs_composer.neg_pairs.items()],
            axis=1)
        neg_data['label'] = np.zeros(neg_data.shape[0], dtype=int)
        pos_data = pd.concat([self.data_in[mod]
             for mod, _ in self.pairs_composer.neg_pairs.items()], axis=1)
        pos_data['label'] = np.ones(pos_data.shape[0], dtype=int)
        self.mid_fusion_data = pd.concat(
            (neg_data.reset_index(drop=True, inplace=False),
             pos_data.reset_index(drop=True, inplace=False)), axis=0).reset_index(drop=True, inplace=False)
        all_pairs_perm = self.mid_fusion_data.index.values[np.random.permutation(self.mid_fusion_data.shape[0])]
        self.mid_fusion_data = self.mid_fusion_data.loc[all_pairs_perm, :].reset_index(drop=True, inplace=False)
        
    def write_pairs(self):
        print("write_pairs")
        chunksize=50000
        chunk_num = ceil(self.mid_fusion_data.shape[0] / chunksize)
        
        if not os.path.exists(self.parquet_out):
            os.makedirs(self.parquet_out)
            
        for chunk in tqdm(range(chunk_num)):
            curr_data_to_write = self.mid_fusion_data.index.values[chunk * chunksize : (chunk + 1) * chunksize]
            fout = os.path.join(self.parquet_out, f"{chunk}.parquet")                
            self.mid_fusion_data.loc[curr_data_to_write, :].to_parquet(fout)
        
    def process(self):
        self.make_data()
        self.write_pairs()
        

import argparse
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import yaml

def parse_yaml_hydra(yaml_in):
    conf_fold, conf_file = os.path.split(yaml_in)
    conf_fold = os.path.split(conf_fold)[1]
    print(conf_fold, conf_file)

    with initialize(config_path=conf_fold):
        # Load a configuration with overrides.
        conf = compose(config_name=conf_file)

    #conf = OmegaConf.to_yaml(conf)
    print(conf)
    return conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--parquet_in', type=str, required=True, help='parquet input data or folder', action='append')
    parser.add_argument(
        '--yaml_in', type=str, required=True, help='yaml corresponding to input')
    parser.add_argument(
        '--checkpoint', type=str, required=True, help='checkpoint corresponding to used net')
    parser.add_argument(
        '--parquet_out', type=str, required=True, help='file used for output')
    parser.add_argument(
        '--col_id', type=str, required=False, default="epk_id", help='id column')
    parser.add_argument(
        '--topn', type=int, required=False, default=30, help='number of top samples for hard negatives')
    parser.add_argument(
        '--mod1_cols', type=str, required=False, help='additional columns', action='append')
    parser.add_argument(
        '--mod2_cols', type=str, required=False, help='additional columns', action='append')
    parser.add_argument(
        '--mod1_name', type=str, required=False)
    parser.add_argument(
        '--mod2_name', type=str, required=False)
    args = parser.parse_args()
    
    cgf = parse_yaml_hydra(args.yaml_in)
    add_cols = {}
    if args.mod1_name:
        add_cols[args.mod1_name] = args.mod1_cols

    if args.mod2_name:
        add_cols[args.mod2_name] = args.mod2_cols
    converter = CrossEncoderConverter(args.parquet_in, cgf, args.checkpoint, 3.0, args.parquet_out, 
        col_id=args.col_id, topn=args.topn, add_mod_cols=add_cols)
    converter.process()            
