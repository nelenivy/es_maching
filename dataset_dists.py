import numpy as np
from sklearn.neighbors import NearestNeighbors
import pyarrow as pa
import torch
import yaml
import pandas as pd
import hydra
import os
import itertools
from tqdm import tqdm
from typing import Union, List, Iterable
from ptls.data_load import IterableChain
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch

import logging
import torch
from matching_bank_x_rmb import M3ColesSupervisedDataset, M3ColesSupervisedIterableDataset
from matching_utils import sort_pair
logger = logging.getLogger(__name__)

 
class ParquetIndexDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_in, i_filters: List[Iterable] = None, index_col=None):
        self.data_in = dd.read_parquet(parquet_in).compute()
        if index_col and index_col in self.data_in.columns:
            self.data_in = self.data_in.set_index(index_col, drop=False)
        
        self.processed_data = {ind: row.to_dict() for ind, row in self.data_in.iterrows()}
        
        if not(i_filters is None):
            post_processor_filter = IterableChain(*i_filters)
            inds = []
            rows = []
            for ind, row in self.processed_data.items():
                inds.append(ind)
                rows.append(row)
                
            rows = [rec for rec in post_processor_filter(r for r in rows)]
            self.processed_data = {ind: row for ind, row in itertools.zip_longest(inds, rows)}
            
        logger.info(f'Loaded {len(self.processed_data)} records')

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        return self.processed_data[item]

def load_model_from_params_and_ckpt(loaded_yaml, checkpoint):
    print("load_model")
    print(loaded_yaml['pl_module'])
    model = hydra.utils.instantiate(loaded_yaml['pl_module'], _convert_="object")
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.to("cuda")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

class DatasetEmbeddingsCalc(object):
    def __init__(self, parquet_in, yaml_in, checkpoint, col_id='epk_id'):
        self.parquet_in = parquet_in
        self.yaml_in = yaml_in
        self.checkpoint = checkpoint
        self.col_id = col_id
        
    def load_yaml(self):
        print("load_yaml")
        with open(self.yaml_in) as f:
            self.conf = yaml.safe_load(f)
        
        if type(self.parquet_in) is str:
            self.conf['data_module']['valid_data']['data']['data_files']['file_path'] = [self.parquet_in]
        elif not (self.parquet_in is None):
            self.conf['data_module']['valid_data']['data']['data_files']['file_path'] = self.parquet_in
        else:
            pass
        
        self.conf['data_module']['valid_data']['_target_'] = "matching_bank_x_rmb.M3ColesSupervisedIterableDataset"
        self.conf['data_module']['valid_data']['cols_classes'] = [self.col_id]
            
    def load_model(self):
        self.model = load_model_from_params_and_ckpt(self.conf, self.checkpoint)
        
    def compute_embeddings(self):
        print("compute_embeddings")
        dm = hydra.utils.instantiate(self.conf['data_module'])
        dl = dm.val_dataloader()
        
        with torch.no_grad():
            self.df = []
            for batch in dl: 
                #print(batch)
                embs = self.model.seq_encoders["data1"](batch[0]["data"].to("cuda"))
                embs = embs.detach().cpu().numpy()
                curr_uids = batch[2]                
                curr_df = pd.DataFrame(index=curr_uids, data=embs, columns = [f'emb_{i}' for i in range(embs.shape[1])])
                self.df.append(curr_df)
            print(batch[0])
            self.df = pd.concat(self.df, axis=0)
            return self.df
        
    def process(self):
        self.load_yaml()
        self.load_model()
        return self.compute_embeddings() 
        
        
class DatasetDistsCalc(object):
    def __init__(self, parquet_in, yaml_in, checkpoint):
        self.parquet_in = parquet_in
        self.yaml_in = yaml_in
        self.checkpoint = checkpoint
        
    def load_yaml(self):
        print("load_yaml")
        with open(self.yaml_in) as f:
            self.conf = yaml.safe_load(f)
        
        if type(self.parquet_in) is str:
            self.conf['data_module']['valid_data']['data']['parquet_in'] = self.parquet_in
            
    def load_model(self):
        self.model = load_model_from_params_and_ckpt(self.conf, self.checkpoint)
            
    def create_dataloader(self):
        dm = hydra.utils.instantiate(self.conf['data_module'])
        self.dl = dm.val_dataloader().dataset
        
    def prepare(self):
        self.load_yaml()
        self.load_model()
        self.create_dataloader()
            
    def calc_dist(self, list_i, list_j):
        if type(list_i) is int:
            list_i = [list_i]
            list_j = [list_j]
            
        batch_i = [self.dl[i] for i in list_i]
        batch_j = [self.dl[j] for j in list_j]
        
        with torch.no_grad():            
            to_model_i = type(self.dl).collate_fn(batch_i)[0]
            to_model_j = type(self.dl).collate_fn(batch_j)[0]
            if type(to_model_i) is tuple:
                to_model_seq = {'data1': to_model_i[0]['data'], 'data2': to_model_j[0]['data']}
                to_model_non_seq = {'data1': to_model_i[1]['data'], 'data2': to_model_j[1]['data']}
            else:
                to_model_seq = {'data1': to_model_i['data'], 'data2': to_model_j['data']}
                to_model_non_seq = None
                
            for k in to_model_seq.keys():
                to_model_seq[k] = to_model_seq[k].to('cuda')
            
            if to_model_non_seq is None:
                res = self.model(to_model_seq)
            else:
                res = self.model((to_model_seq, to_model_non_seq))
            #print(res)
            proba = res.cpu().numpy().ravel()
            dist = 1.0 - proba
            return dist     
        

class DatasetEmbsAndDistsCalc(object):
    def __init__(self, embs, dists_calculator=None):
        self.df = embs
        self.dists_calculator = dists_calculator

    def nearest_neigbours(self):
        self.N_NEIGHBOURS = 100
        knn = NearestNeighbors(n_neighbors=self.N_NEIGHBOURS, metric='cosine')
        knn = knn.fit(self.df.values)
        self.dists, self.neighb_inds = knn.kneighbors(X=self.df.values)

    def make_pairs(self, batch_size=256):
        inf_num_pairs = []
        self.pairs_k = {1: [], 5: [], 10: [], 20: [], 30: [], 40: [], 50: [], 100: []}
        pairs_set_k = {1: set(), 5: set(), 10: set(), 20: set(), 30: set(), 40: set(), 50: set(), 100: set()}
        #make pairs from nearest neighbors
        all_pairs = []
        for i in tqdm(range(self.df.shape[0])):
            curr_elem = self.df.index.values[i]
            for ind in range(self.N_NEIGHBOURS):
                curr_neighb_ind = self.neighb_inds[i, ind]
                if curr_neighb_ind == i:
                    continue
                curr_neighb = self.df.index.values[curr_neighb_ind]
                curr_dist = self.dists[i, ind]                
                pair = sort_pair([curr_elem, curr_neighb])
                all_pairs.append((pair[0], pair[1], curr_dist, ind))
        
        print(all_pairs[:100])
        if self.dists_calculator:
            #recalculate dist 
            new_all_pairs = []
            for start_i in tqdm(range(0, len(all_pairs), batch_size)):
                curr_pairs = all_pairs[start_i : start_i + batch_size]
                list_i = [p[0] for p in curr_pairs]
                list_j = [p[1] for p in curr_pairs]
                curr_dists = self.dists_calculator.calc_dist(list_i, list_j)
                new_all_pairs += [(p1, p2, d, curr_p[3]) for p1, p2, d, curr_p in zip(list_i, list_j, curr_dists, curr_pairs)]   
            all_pairs = new_all_pairs
            
        for p1, p2, curr_dist, ind in all_pairs:
            pair = (p1, p2)
            for k in self.pairs_k.keys():
                if ind <= k and not (pair in pairs_set_k[k]):
                    self.pairs_k[k].append((pair[0], pair[1], curr_dist))
                    pairs_set_k[k].add(pair)  
                        
    def process(self, batch_size=256):
        self.nearest_neigbours()
        self.make_pairs(batch_size=batch_size)
        return self.pairs_k

