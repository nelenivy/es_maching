import pyarrow.parquet as pq
import pandas as pd
import torch
import numpy as np
import math
from collections import defaultdict

from ptls.data_load.datasets import ParquetDataset, ParquetFiles
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load import IterableChain
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch

from functools import reduce
from operator import iadd

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit
from ptls.frames.coles.metric import metric_recall_top_K, outer_cosine_similarity, outer_pairwise_distance

from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from matching_bank_x_rmb import M3ColesDatasetBase, M3ColesSupervisedDatasetMixin, PrepareOnePairMixin, PrepareOnePairSupervisedMixin, split_dict_of_lists_by_one
from scipy.stats import lognorm, norm, bernoulli
from time_proc_matching import convert_to_datetime_seconds
from statistics import median

def SplitByFeat(feat):
    #return split indices
    splits = defaultdict(list)
    seq_len = len(feat)
    
    for i in range(len(feat)):
        curr_id = feat[i]
        splits[curr_id].append(i)
        
    return splits

class CrossDomainGenerator:
    def __init__(self, mean_succ=10.0, std_succ=5.0, 
                 mean_fail=None, std_fail=None,
                 success_proba=0.5):
        self.gen = norm()#lognorm(s=std, scale=mean)
        if mean_fail is None:
            mean_fail = mean_succ
            
        if std_fail is None:
            std_fail = std_succ
        self.mean = [mean_fail, mean_succ]
        self.std = [std_fail, std_succ]
        self.success_proba = success_proba
        
        
    def __call__(self, time_seq):
        ts = convert_to_datetime_seconds(time_seq)
        #print(time_seq, ts, day)
        seq_len = ts.shape[0]
        curr_pos = 0
        segmentation = np.zeros(seq_len, np.int32)
        curr_segm = 0
        seconds_in_day = 24.0 * 60.0 * 60.0
        
        while curr_pos < seq_len:
            #print("+++++++++++++++++++++++++++++++")
            curr_days_num = -1.0
            
            success_cross_domain = bernoulli.rvs(self.success_proba)
            
            while curr_days_num < 0.01:                
                curr_days_num = self.mean[success_cross_domain] + self.gen.rvs() * self.std[success_cross_domain]            
            
            for next_pos in range(curr_pos, seq_len):
                curr_diff = np.abs((ts[next_pos] - ts[curr_pos]).astype('timedelta64[s]').view('int64'))
                curr_diff = curr_diff.astype(float) / seconds_in_day
                
                if not success_cross_domain:
                    segmentation[next_pos] = curr_segm
                    curr_segm += 1
                #print(ts[next_pos], ts[curr_pos], curr_diff, curr_days_num)
                if curr_diff >= curr_days_num:
                    break
            
            if next_pos == seq_len:
                next_pos -= 1
             
            if success_cross_domain:
                segmentation[curr_pos : next_pos + 1] = curr_segm
                curr_segm += 1
            curr_pos = next_pos + 1
            
        
        return segmentation

def CombineSplits(features_seqs, first_party_col, time_col, cross_domain_generator):
    first_party = np.array(features_seqs[first_party_col])
    time_feat = np.array(features_seqs[time_col])
    seq_len = len(list(features_seqs.values())[0])
    cross_domain = cross_domain_generator(time_feat)  
    #print(cross_domain)
    cross_domains_num = np.max(cross_domain) + 1   
    
    #print(len(first_party), len(first_parties))
    #print(len(cross_domain), cross_domains_num)
    
    id_to_components = dict()
    components = []
    nums_to_delete = set()
    
    for i in range(seq_len):
        curr_first_party = (first_party[i], "1st_party")
        curr_cross_domain = (cross_domain[i], "cross_domain")
        first_party_in = curr_first_party in id_to_components
        cross_domain_in = curr_cross_domain in id_to_components
        
        if first_party_in and not cross_domain_in:
            set_num = id_to_components[curr_first_party]
            components[set_num].add(curr_cross_domain)
            id_to_components[curr_cross_domain] = set_num
        elif cross_domain_in and not first_party_in:
            set_num = id_to_components[curr_cross_domain]
            components[set_num].add(curr_first_party)
            id_to_components[curr_first_party] = set_num
        elif cross_domain_in and first_party_in:
            set_num_1 = id_to_components[curr_cross_domain]
            set_num_2 = id_to_components[curr_first_party]
            
            if set_num_1 != set_num_2:
                components[set_num_1] = components[set_num_1].union(components[set_num_2])
                
                for k in id_to_components.keys():
                    #rename all occurences
                    if id_to_components[k] == set_num_2:
                        id_to_components[k] = set_num_1
                nums_to_delete.add(set_num_2)
        else:
            new_set = set((curr_first_party, curr_cross_domain))
            id_to_components[curr_first_party] = len(components)
            id_to_components[curr_cross_domain] = len(components)
            components.append(new_set)

            
    first_party_splits = SplitByFeat(first_party) 
    cross_domain_splits = SplitByFeat(cross_domain) 
    
    return_splits = []
    
    for i in range(len(components)):
        if i in nums_to_delete:
            continue
        comp = components[i]
        all_ids = set()
        for c, t in comp:
            curr_ids = first_party_splits[c] if (t == "1st_party") else cross_domain_splits[c]
            all_ids = all_ids.union(set(curr_ids))
            
        all_ids = sorted(list(all_ids))
        return_splits.append(all_ids)

    return return_splits
        

class M3ColesCrossDomainDatasetBase(M3ColesDatasetBase):
    """
    Multi-Modal Matching
    Dataset for ptls.frames.coles.CoLESModule
    Parameters
    ----------
    data:
        source data with feature dicts
    splitter:
        object from from `ptls.frames.coles.split_strategy`.
        Used to split original sequence into subsequences which are samples from one client.
    col_time:
        column name with event_time
    """

    def __init__(self,
                 data,
                 splitter: AbsSplit,
                 first_party_col='site1',
                 cross_domain_generator=None,
                 col_time='event_time',
                 mod_names=('bank', 'rmb'),
                 split_mod=None,
                 min_len=10,
                 max_comps=7,
                 log_n_steps=100*256,
                 yield_by_one=False,
                 *args, **kwargs):
        super().__init__(data,
                 splitter=splitter,
                 col_time=col_time,
                 mod_names=mod_names,
                 yield_by_one=yield_by_one,
                         *args, **kwargs)  # required for mixin class

        self.first_party_col = first_party_col
        self.min_len = min_len
        self.max_comps = max_comps
        self.split_mod = split_mod
        
        if cross_domain_generator is None:
            cross_domain_generator = CrossDomainGenerator(7.0, 10.0, 0.5)
            
        self.cross_domain_generator = cross_domain_generator 
        self.i = 0
        self.log_n_steps = log_n_steps
        
        if self.log_n_steps > 0:
            self.uncut_segms = []
            self.uncut_elems = []
            self.deleted_small_segms = []
            self.deleted_small_elems = []
            self.deleted_segms = []
            self.deleted_elems = []
            self.num_non_zero_users = 0
            
    def get_splits(self, modalities):        
        modal_splits_cross_domain = []
        seq_to_split = modalities[self.split_mod]  
        splits_ids_uncut = CombineSplits(seq_to_split, self.first_party_col, self.col_time, self.cross_domain_generator)

        splits_ids = [idx for idx in splits_ids_uncut if len(idx) >= self.min_len]
        
        if self.log_n_steps > 0:
            #only for logging
            self.deleted_small_segms.append((len(splits_ids_uncut) - len(splits_ids)) / float(len(splits_ids_uncut)))
            
            uncut_all_len = len([e for elem in splits_ids_uncut for e in elem])
            cut_all_len_small = len([e for elem in splits_ids for e in elem])
            
            self.deleted_small_elems.append((uncut_all_len - cut_all_len_small)  / float(uncut_all_len))
            
        if len(splits_ids) > self.max_comps:
            splits_ids = splits_ids[: self.max_comps]
        
        #only for logging
        if self.log_n_steps > 0:
            self.num_non_zero_users += len(splits_ids) > 0
            cut_all_len = len([e for elem in splits_ids for e in elem])
            
            self.deleted_elems.append(uncut_all_len - cut_all_len)
            self.deleted_segms.append(len(splits_ids_uncut) - len(splits_ids))
            
            self.uncut_elems.append(self.deleted_elems[-1] / float(uncut_all_len))
            self.uncut_segms.append(self.deleted_segms[-1] / float(len(splits_ids_uncut)))
            self.i += 1
            
            if self.i % self.log_n_steps == 0:
                for l, t in [(self.uncut_segms, "all segms"), (self.uncut_elems, "all events"), 
                             (self.deleted_elems, "deleted events"), (self.deleted_segms, "deleted segms"),
                            (self.deleted_small_elems, "deleted small events"), (self.deleted_small_segms, "deleted small segms")]:                             
                    print("mean " + t, sum(l) / self.i,
                          "med " + t, median(l))
                print("non zero users rate", self.num_non_zero_users / float(self.i))
                    
        cross_domain_mods = {}

        for mod_name, mod_seq in modalities.items():
            curr_mod_data = []
            if mod_name == self.split_mod:
                curr_mod_data = self.apply_splits_to_feat_dict(seq_to_split, splits_ids)
            else:
                curr_mod_data = [mod_seq] * len(splits_ids)
            
            mod_splits = []
            for elem in curr_mod_data:
                splited_by_spliter = self.get_one_modality_time_split(elem)
                mod_splits += splited_by_spliter
            
            cross_domain_mods[mod_name] = mod_splits
                
        return cross_domain_mods

class M3ColesCrossDomainDataset(M3ColesCrossDomainDatasetBase, PrepareOnePairMixin):
    pass

class M3ColesCrossDomainIterableDataset(M3ColesCrossDomainDataset, torch.utils.data.IterableDataset):
    pass


class M3ColesCrossDomainSupervisedDataset(M3ColesCrossDomainDatasetBase, M3ColesSupervisedDatasetMixin):
    @staticmethod
    def collate_fn(batch):
        return M3ColesSupervisedDatasetMixin.collate_fn(batch)
    
    def split_one_by_one(self, prepared_pairs):
        dict_of_lists, classes = prepared_pairs
    
        list_of_feats = split_dict_of_lists_by_one(dict_of_lists)
        to_return = [(l, [c]) for l ,c in zip(list_of_feats, classes)]
        return to_return    


class M3ColesCrossDomainSupervisedIterableDataset(M3ColesCrossDomainSupervisedDataset, torch.utils.data.IterableDataset):
    pass