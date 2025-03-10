import numpy as np
import torch
import yaml
import pandas as pd
import hydra
import os
import itertools
from tqdm import tqdm
from typing import Union, List, Iterable
import logging
import torch
from collections import defaultdict
import omegaconf
logger = logging.getLogger(__name__)
from dataset_dists import load_model_from_params_and_ckpt
from m3_utils import batch_nearest_neighbors
from matching_metrics import PairsClassificationQuality
from matching_bank_x_rmb import M3ColesDatasetBase
from omegaconf import OmegaConf, open_dict

def resolve_interpolations(cfg):
    print(type(cfg), isinstance(cfg, dict))
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            print(k, v, type(v))
            if isinstance(v, str) and "${" in v:
                # Simple resolution logic; may need more sophisticated parsing for nested cases
                var_name = v.replace("${", "").replace("}", "")
                print(var_name)
                if var_name in cfg:
                    cfg[k] = cfg[var_name]
            elif isinstance(v, (dict, list)):
                resolve_interpolations(v)
    elif isinstance(cfg, list):
        for item in cfg:
            resolve_interpolations(item)

class DataLoaderWithColumns:
    def __init__(self, parquet_in, yaml_in, dataset_class, col_id='epk_id'):
        self.parquet_in = parquet_in
        self.yaml_in = yaml_in
        self.col_id = col_id
        self.dataset_class = dataset_class
        
    def load_yaml(self):
        print("load_yaml")
        #with open(self.yaml_in) as f:
        #self.conf = yaml.safe_load(f)
        #self.conf = omegaconf.OmegaConf.create(self.conf)
        self.conf = self.yaml_in
        print(self.conf)
        resolve_interpolations(self.conf)
        print(self.conf)
        if type(self.parquet_in) is str:
            self.conf['data_module']['valid_data']['data']['data_files']['file_path'] = [self.parquet_in]
        elif not (self.parquet_in is None):
            self.conf['data_module']['valid_data']['data']['data_files']['file_path'] = self.parquet_in
        else:
            pass
        
        print('conf', self.conf['data_module']['valid_data'])
        self.conf['data_module']['valid_data']['_target_'] = self.dataset_class#M3ColesSupervisedIterableDataset
        print('conf', self.conf['data_module']['valid_data'])
        with open_dict(self.conf):
            #cfg.foo.cc = 30
            self.conf['data_module']['valid_data']['cols_classes'] = [self.col_id]

        
        filters = self.conf['data_module']['valid_data']['data']['i_filters']
        print(filters)
        new_filters = []
        feature_filters = ["ptls.data_load.iterable_processing.feature_filter.FeatureFilter",
            'ptls.data_load.iterable_processing.FeatureFilter']
        for i in range(len(filters)):
            print(filters[i])

            print(filters[i]["_target_"], type(filters[i]["_target_"]), 
                  filters[i]["_target_"] in feature_filters,
                 filters[i]["_target_"] == "ptls.data_load.iterable_processing.iterable_shuffle.IterableShuffle")
            
            if filters[i]["_target_"] in feature_filters:
                with open_dict(filters):
                    filters[i]["keep_feature_names"] = [self.col_id]
                    filters[i]["drop_non_iterable"] = False
                
                    if "drop_feature_names" in filters[i]:
                        drop_feature_names = []
                        for feat_name in filters[i]["drop_feature_names"]:
                            if feat_name != self.col_id:
                                    drop_feature_names.append(feat_name)
                                    
                        print(filters[i]["drop_feature_names"], drop_feature_names)
                        filters[i]["drop_feature_names"] = drop_feature_names
                
            if filters[i]["_target_"] == "ptls.data_load.iterable_processing.iterable_shuffle.IterableShuffle":
                continue
            
            new_filters.append(filters[i])                
                
        self.conf['data_module']['valid_data']['data']['i_filters'] = new_filters        
        self.conf['data_module']['valid_data']['data']['_target_'] =  "parquet_shuffle_dataset.NoThreadsParquetDataset"  
        
    def create_dataloader(self):
        dm = hydra.utils.instantiate(self.conf['data_module'])
        self.dl = dm.val_dataloader().dataset
    
    def read_data(self):
        self.id_to_feats = {}
        for feats, ids in self.dl:
            assert(len(ids) == 1)
            self.id_to_feats[ids[0]] = feats        
        print(list(self.id_to_feats.items())[:2])   
        print(f'Loaded {len(self.id_to_feats)} records')
        
    def prepare(self):
        self.load_yaml()
        self.create_dataloader()
        self.read_data()
        
class M3DatasetEmbeddingsCalc:
    def __init__(self, parquet_in, yaml_in, checkpoint, col_id='epk_id', keep_on_gpu=True):
        self.parquet_in = parquet_in
        self.yaml_in = yaml_in
        self.checkpoint = checkpoint
        self.col_id = col_id
        self.keep_on_gpu = keep_on_gpu  
        
    def load_yaml(self):        
        print("load_yaml")
        data_loader = DataLoaderWithColumns(self.parquet_in, self.yaml_in, 
                              "matching_bank_x_rmb.M3ColesSupervisedIterableDataset", col_id=self.col_id)
        data_loader.load_yaml()
        self.conf = data_loader.conf
            
    def load_model(self):
        self.model = load_model_from_params_and_ckpt(self.conf, self.checkpoint)
        
    def compute_embeddings(self):
        print("compute_embeddings")
        dm = hydra.utils.instantiate(self.conf['data_module'])
        dl = dm.val_dataloader()
        self.embs = defaultdict(list)
        self.uids = []
        
        with torch.no_grad():        
            for i, batch in enumerate(dl): 
                if i % 10 ==0:
                    print(i)
                feats, entity_labels, curr_uids = batch
                feats = {k: elem.to("cuda") for k, elem in feats.items()}
                curr_embs, _ = self.model.shared_step(feats, "")
                
                if not self.keep_on_gpu:
                    curr_embs = {k: emb.detach().cpu().numpy() for k, emb in curr_embs.items()}
                    
                self.uids.append(curr_uids)
                for k, emb in curr_embs.items():
                    self.embs[k].append(emb)
                    
                #print(list(self.embs.keys()))
                
        for k in self.embs.keys():
            if self.keep_on_gpu:
                self.embs[k] = torch.concat(self.embs[k], dim=0)
            else:
                self.embs[k] = np.concatenate(self.embs[k], axis=0)
                
        self.uids = np.concatenate(self.uids, axis=0)
            
        return self.embs, self.uids
    
            
    def process(self):
        self.load_yaml()
        self.load_model()
        return self.compute_embeddings() 
    
class M3DatasetDistsCalc(object):
    def __init__(self, parquet_in, yaml_in, checkpoint, col_id=None):
        self.parquet_in = parquet_in
        self.yaml_in = yaml_in
        self.checkpoint = checkpoint
        self.col_id = col_id
        self.data_loader = DataLoaderWithColumns(self.parquet_in, self.yaml_in, 
                              "matching_bank_x_rmb.M3ColesSupervisedIterableDataset", col_id=self.col_id)

    def prepare(self):
        self.data_loader.prepare()
        self.model = load_model_from_params_and_ckpt(self.data_loader.conf, self.checkpoint)
            
    def calc_dist(self, mod_to_ind_dict):
        batch_feats = []
        for mod_to_ind in mod_to_ind_dict:
            batch_feats.append({mod: self.data_loader.id_to_feats[i][mod] for mod, i in mod_to_ind.items()})
        
        with torch.no_grad():            
            to_model_seq = M3ColesDatasetBase.collate_fn(batch_feats)[0]
                
            for k in to_model_seq.keys():
                to_model_seq[k] = to_model_seq[k].to('cuda')            
            
            res = self.model(to_model_seq, use_sigmoid=True)
            proba = res.cpu().numpy().ravel()
            dist = 1.0 - proba
            return dist     
        
class M3DatasetEmbsAndDistsCalc(object):
    def __init__(self, embs, uids, dists_calculator=None):
        self.embs = embs
        self.uids = uids
        self.dists_calculator = dists_calculator
        print('uids', self.uids)
        
    def nearest_neighbors(self, n_neighbs=30):
        print("nearest_neighbors")
        self.N_NEIGHBOURS = n_neighbs
        embs_to_nn = [emb for k, emb in self.embs.items()]
        self.neighb_inds, self.dists = batch_nearest_neighbors(embs_to_nn[0], embs_to_nn[1], K=self.N_NEIGHBOURS, return_dists=True, batch_size=256)
        self.neighb_inds = {k: neighb_ind for (k, emb), neighb_ind in zip(self.embs.items(), self.neighb_inds)}
        self.dists = {k: dist_mod for (k, emb), dist_mod in zip(self.embs.items(), self.dists)}
        print(self.neighb_inds)
        print(self.dists)
        
    def make_pairs(self, batch_size=256, filter_by_another_mod=True):
        inf_num_pairs = []
        
        #make pairs from nearest neighbors
        
        mod_names = list(self.embs.keys())
        self.mod_all_pairs = [{}, {}]
        
        for mod_num, mod in enumerate(mod_names):
            other_mod_num = int(not(mod_num))
            
            for i in range(self.uids.shape[0]):
                curr_elem = self.uids[i]
                for ind in range(self.N_NEIGHBOURS):
                    curr_neighb_ind = self.neighb_inds[mod][i, ind]      
                    curr_neighb = self.uids[curr_neighb_ind]
                    
                    #if curr_neighb == curr_elem:
                    #    continue
                        
                    curr_dist = self.dists[mod][i, ind]   
                    pair = (curr_elem, curr_neighb) if mod_num == 0 else (curr_neighb, curr_elem)
                    self.mod_all_pairs[mod_num][pair] = [curr_dist, ind]
          
        all_pairs = {}
        if filter_by_another_mod:
            for pair, info in self.mod_all_pairs[0].items():
                if pair in self.mod_all_pairs[1]:
                    all_pairs[pair] = (info[0], max(info[1],  self.mod_all_pairs[1][pair][1]))
        else:
            for mod_num in (0, 1):
                for pair, info in self.mod_all_pairs[mod_num].items():    
                    num = info[1]
                    if pair in all_pairs:
                        num = max(num, all_pairs[pair][1])
                    all_pairs[pair] = (info[0], num)       

        all_pairs = list(all_pairs.items())                
        print(all_pairs[:100])

        if self.dists_calculator:
            #recalculate dist 
            new_all_pairs = []
            for start_i in tqdm(range(0, len(all_pairs), batch_size)):
                curr_pairs = all_pairs[start_i : start_i + batch_size]
                list_i = [p[0][0] for p in curr_pairs]
                list_j = [p[0][1] for p in curr_pairs]
                
                curr_dists = self.dists_calculator.calc_dist([{mod_names[0]:i, mod_names[1]:j} for i, j in zip(list_i, list_j)])
                #print([p[2] for p in curr_pairs])
                #print(curr_dists)
                #print("--------------")
                new_all_pairs += [((p1, p2), (d, curr_p[1][1])) for p1, p2, d, curr_p in zip(list_i, list_j, curr_dists, curr_pairs)]   
            all_pairs = new_all_pairs
        
        threshes = [1, 5] + list(range(10, 50, 10)) + list(range(50, 500, 50))
        self.pairs_k = {t:[] for t in threshes if t <= self.N_NEIGHBOURS}
        self.pairs_set_k = {t:set() for t in threshes if t <= self.N_NEIGHBOURS}
        
        for (p1, p2), (curr_dist, ind) in all_pairs:
            pair = (p1, p2)
            for k in self.pairs_k.keys():
                if ind < k and not (pair in self.pairs_set_k[k]):
                    
                    self.pairs_k[k].append((pair[0], pair[1], curr_dist))
                    self.pairs_set_k[k].add(pair)  
                        
    def process(self, batch_size=128, filter_by_another_mod=True, n_neighbs=30):
        self.nearest_neighbors(n_neighbs=n_neighbs)
        self.make_pairs(batch_size=batch_size, filter_by_another_mod=filter_by_another_mod)
        return self.pairs_k

def calc_m3_quality(parquet_in, yaml_embs, checkpoint_embs, yaml_early_fusion=None, checkpoint_early_fusion=None, 
                    col_id='epk_id', folder_out=None, chunks_size=-1, samples_num=-1,
                    write_matched=False, no_quality=False):
    embeddings_computer = M3DatasetEmbeddingsCalc(parquet_in, yaml_embs, checkpoint_embs, col_id=col_id)
    embs, uids = embeddings_computer.process()
    del embeddings_computer
    
    if not(yaml_early_fusion is None):
        reranker = M3DatasetDistsCalc(parquet_in, yaml_early_fusion, checkpoint_early_fusion, col_id=col_id)
        reranker.prepare()
    else:
        reranker = None
              
    """
    chunk_embs = []
    chunk_uids = []

    if (chunks_size > 0) and (samples_num > 0):
                            for _ in range(samples_num):
                                          curr_ids = np.random.choice(np.arange(embs.shape[0]), size=samples_num, replace=False)
                                          chunk_embs.append(embs[curr_ids, :])
                                          chunk_uids.append(uids[curr_ids])
    else:
        chunk_embs = [embs]
        chunk_uids = [uids]
                            
    for chunk_num in range(len(chunk_embs)):
                            uids = chunk_uids[chunk_num]
                            embs = chunk_embs[chunk_num]
    """
    retriever_and_reranker = M3DatasetEmbsAndDistsCalc(embs, uids, reranker)
    pairs_k = retriever_and_reranker.process(batch_size=128, n_neighbs=100)
    golden_set = set([(uid, uid) for uid in uids])
    all_pairs_num = uids.shape[0] ** 2
    
    if write_matched:        
        top_k_for_elem = [defaultdict(list), defaultdict(list)]
        max_neghb_num = max(pairs_k.keys())
        
        for pair_with_d in pairs_k[max_neghb_num]:
            for mod_num in (0, 1):
                top_k_for_elem[mod_num][pair_with_d[mod_num]].append((pair_with_d[1 - mod_num], pair_with_d[2]))
        
        ids_mod = []
        ids_to_num_mod = []
        for mod_num in (0, 1):
            ids_mod.append(list(top_k_for_elem[mod_num].keys()))
            ids_to_num_mod.append({i: num for num, i in enumerate(ids_mod[-1])})
            ids_to_num_mod[-1][''] = -1
           
        print(list(top_k_for_elem[0].items())[:100])
        print(len(top_k_for_elem[0].items()))
        print(ids_mod[0][:10])
              
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)
        
        for mod_num in (0, 1):
            dists = []
            nids = []
            print(1)
            this_mod_ids = ids_mod[mod_num]
            print(2)
            other_mod_ids_to_num = ids_to_num_mod[1 - mod_num]
            print(3)
            
            for p1 in tqdm(top_k_for_elem[mod_num].keys()):
                curr_neighbs = sorted(top_k_for_elem[mod_num][p1], key=lambda elem: elem[1])
                
                if len(curr_neighbs) < max_neghb_num:
                    curr_neighbs += [("", -1)] * (max_neghb_num - len(curr_neighbs))
                
                dists.append(np.array([d for p, d in curr_neighbs])[None, :])
                nids.append(np.array([other_mod_ids_to_num[p] for p, d in curr_neighbs])[None, :]) 
            
            print(4)
            this_mod_ids = np.array(this_mod_ids)
            print(5)
            dists = np.vstack(dists)#pd.DataFrame(data=np.vstack(dists), index=ids, columns=np.array([f'd_{i}' for i in range(max_neghb_num)]))
            print(6)
            nids = np.vstack(nids)#pd.DataFrame(data=np.vstack(nids), index=ids, columns=np.array([f'id_{i}' for i in range(max_neghb_num)]))
            print(7)
            np.savez(os.path.join(folder_out, "neighb_" + str(mod_num) + ".npz"), dists=dists, nids=nids, ids=this_mod_ids)
            #neighbs_frame  = pd.concat([dists, nids], axis=1)
            #neighbs_frame.to_parquet(os.path.join(folder_out, "neighb_" + str(mod_num) + ".parquet"))
                
    if not no_quality:
        for k in pairs_k.keys():
            quality_calc = PairsClassificationQuality()
            res = quality_calc.process(pairs_k[k], golden_set, all_pairs_num, prefix_length=0, sort_ids_in_pair=False)
            print(k, res)
            for j in range(2):
                maxi = np.argmax(quality_calc.f1_top_k[j])
                print(quality_calc.prec_top_k[j][maxi],
                      quality_calc.recalls[maxi],
                      np.max(quality_calc.f1_top_k[j]), np.argmax(quality_calc.f1_top_k[j]))
                      
            if folder_out:
                if not os.path.exists(folder_out):
                    os.makedirs(folder_out)
                    
                out_quality = pd.DataFrame(data={"prec": np.array(quality_calc.precs), 
                "recall": np.array(quality_calc.recalls),  
                "f1": np.array(quality_calc.f1s),  
                "tps": np.array(quality_calc.tps), 
                "fps": np.array(quality_calc.fps),
                "threshes": np.array(quality_calc.threshes),
                "prec_top_k_1": np.array(quality_calc.prec_top_k[0]),
                "prec_top_k_2": np.array(quality_calc.prec_top_k[1]),
                "f1_top_k_1": np.array(quality_calc.f1_top_k[0]),
                "f1_top_k_2": np.array(quality_calc.f1_top_k[1])})
                out_quality.to_csv(os.path.join(folder_out, f"quality_{k}.csv"))
                                           
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--parquet_in', type=str, required=True, help='parquiet input data or folder', action='append')
    parser.add_argument(
        '--yaml_embs', type=str, required=True, help='yaml for embeddings')
    parser.add_argument(
        '--checkpoint_embs', type=str, required=True, help='checkpoint for embeddings')
    parser.add_argument(
        '--yaml_reranker', type=str, required=False, default=None, help='yaml for reranker')
    parser.add_argument(
        '--checkpoint_reranker', type=str, required=False, default=None, help='checkpoint for reranker')
    parser.add_argument(
        '--folder_out', type=str, required=True, help='folder used for output')
    parser.add_argument(
        '--col_id', type=str, required=False, default="epk_id", help='id column')
    parser.add_argument(
        '--chunk_size', type=int, required=False, default=-1, 
          help='chunk_size used for calculating metric if it is used')
    parser.add_argument(
        '--samples_num', type=int, required=False, default=-1, 
          help='samples number used for calculating metric if it is used')
    parser.add_argument(
        '--write_matched', action="store_true")
    parser.add_argument(
        '--no_quality', action="store_true")



    args = parser.parse_args()
    print(args.col_id)
    calc_m3_quality(args.parquet_in, args.yaml_embs, args.checkpoint_embs, yaml_early_fusion=args.yaml_reranker, checkpoint_early_fusion=args.checkpoint_reranker, 
                    col_id=args.col_id, folder_out=args.folder_out, 
                    chunks_size=args.chunk_size, samples_num=args.samples_num,
                    write_matched=args.write_matched, no_quality=args.no_quality)
    
