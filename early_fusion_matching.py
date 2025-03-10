import numpy as np
import torchmetrics
import sklearn
from sklearn.metrics import roc_auc_score
import torch
import pandas as pd
import os
import itertools
from tqdm import tqdm
from typing import Union, List, Iterable
from ptls.data_load import IterableChain
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch
from scipy.stats.stats import pearsonr
from scipy import spatial

import logging
from ptls.frames.coles import MultiModalSortTimeSeqEncoderContainer
from matching_bank_x_rmb import M3ColesSupervisedDataset

import math
from collections import defaultdict

from ptls.data_load.datasets import ParquetDataset, ParquetFiles
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset

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
from ptls.nn.trx_encoder.noisy_embedding import NoisyEmbedding
from ptls.data_load.padded_batch import PaddedBatch
from resample import resample, add_space_and_session_tokens, get_tensor_by_indices
from x_transformer import XTransformerEncoder
from warmup_mixin import WarmupMixin
logger = logging.getLogger(__name__)

#a helper class used to plug parquet data into multimodal interface
class ParquetToDict(torch.utils.data.Dataset):
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
        self.out_data = defaultdict(dict)
        
        for ind, row in self.processed_data.items():
            for col, val in row.items():
                self.out_data[col][ind] = val
                
        self.out_data = {col: pd.Series(vals) for col, vals in self.out_data.items()}
        logger.info(f'Loaded {len(list(self.out_data.values())[0])} records')          
        

    def __len__(self):
        return len(self.out_data)

    def __getitem__(self, item):
        return self.out_data[item]
    
    def items(self):
        return self.out_data.items()
    
    def values(self):
        return self.out_data.values()
        
    def keys(self):
        return self.out_data.keys()
    
    def __iter__(self):
        for elem in self.out_data:
            yield elem

def get_distribution(nums, max_num):
    count = defaultdict(float)
    for num in nums:
        count[num]+=1.0
    return np.array([count[_] for _ in range(max_num)])

def get_24hr_distribution(nums):
    return get_distribution(nums, 24)

def get_24hr_weekly_distribution(nums):
    return get_distribution(nums, 24 * 7)


class EarlyFusionM3SupervisedDataset(M3ColesSupervisedDataset):
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
                 col_time='event_time',
                 mod_names=('bank', 'rmb'),
                 add_time_hist=False,
                 cols_classes = None,
                 *args, **kwargs):
        super().__init__(data=data,
                 splitter=splitter,
                 col_time=col_time,
                 mod_names=mod_names,
                 cols_classes=cols_classes,
                  *args, **kwargs)  # required for mixin class
        
        self.add_time_hist = add_time_hist

    def get_splits(self, modalities):
        return_splits = {}
        
        if self.add_time_hist:
            return_non_seq_feats = defaultdict(list)
            
        for mod_name, feature_arrays in modalities.items():
            local_date = feature_arrays[self.col_time]
            indexes = self.splitter.split(local_date)
            
            if self.add_time_hist:
                for ix in indexes:
                    hours = feature_arrays['hour'][ix].numpy()
                    week_hours = feature_arrays['weekday'][ix].numpy()
                    hour_dist = get_24hr_distribution(hours)
                    weekday_dist = get_24hr_weekly_distribution(week_hours)                 
                    return_non_seq_feats[mod_name].append({"hour_dist": hour_dist, "weekday_dist": weekday_dist})
                                                          
            return_splits[mod_name] = [
                {k: v[ix] for k, v in feature_arrays.items() if (self.is_seq_feature(k, v))} 
                                       for ix in indexes]
            
        to_return = (return_splits, return_non_seq_feats) if self.add_time_hist else return_splits
        return to_return
    
    def modality_split(self, feature_arrays):
        modalities = {}
        for mod_name in self.mod_names:
            curr_mod_feature_dict = {k.replace(mod_name + '_', ''):v for k,v in feature_arrays.items() if k.startswith(mod_name)}
            modalities[mod_name] = curr_mod_feature_dict
        return modalities
      
    @staticmethod
    def collate_fn(batch):
        target_labels = []
      
        for seq, labels in batch:
            target_labels += labels
        #print(type(batch), type(batch[0]), type(batch[0][0]))
        
        use_time_hist = type(batch[0][0]) is tuple
        padded_batch = dict()        
        
        if use_time_hist:
            mod_names = batch[0][0][0].keys()
            batch_seq_feats = {mod: [b[0][0][mod] for b in batch] for mod in mod_names}
            batch_non_seq_feats = {mod:
                [b[0][1][mod] for b in batch]
                         for mod in mod_names}
        else:
            mod_names = batch[0][0].keys()
            batch_seq_feats = {mod: [b[0][mod] for b in batch] for mod in mod_names}

            
        padded_batch = {mod: collate_feature_dict(reduce(iadd, batch_seq_feats[mod])) for mod in mod_names}
        
        if use_time_hist:
            batch_non_seq_feats = {mod: reduce(iadd, batch_non_seq_feats[mod]) for mod in mod_names}
            return (padded_batch, batch_non_seq_feats), target_labels
        else:
            return padded_batch, target_labels


class EarlyFusionM3SupervisedIterableDataset(EarlyFusionM3SupervisedDataset, torch.utils.data.IterableDataset):
    pass

def pearson_correlation(x,y):
    return pearsonr(x,y)[0]

def cosine_distance(x,y):
    return spatial.distance.cosine(x, y)

class DummyTrxEncoder:
    def forward(x):
        pass
    
class M3SortTimeSeqEncoderContainerSameMod(MultiModalSortTimeSeqEncoderContainer):
    def __init__(self,
                 trx_encoders,
                 seq_encoder_cls, 
                 input_size,
                 is_reduce_sequence=True,
                 col_time='event_time',
                 **seq_encoder_params
                ):
        
        for k in trx_encoders.keys():
            if type(trx_encoders[k]) is str:
                trx_encoders[k] = trx_encoders[trx_encoders[k]]

                
        super().__init__(trx_encoders=trx_encoders,
                 seq_encoder_cls=seq_encoder_cls, 
                 input_size=input_size,
                 is_reduce_sequence=is_reduce_sequence,
                 col_time=col_time,
                 **seq_encoder_params)
        
    #the modification is that time is returned
    def merge_by_time(self, x):
        device = list(x.values())[1][0].device
        batch, batch_time = torch.tensor([], device=device), torch.tensor([], device=device)
        all_emb_idx = torch.tensor([], device=device)

        for i, source_batch in x.values():
            if source_batch[0] != 'None':
                batch = torch.cat((batch, source_batch[1].payload), dim=1)
                batch_time = torch.cat((batch_time, source_batch[0]), dim=1)
                batch_idx = torch.ones(
                    (batch_size, source_batch[1].payload.shape[1]), dtype=int, device=device) \
                    * i
                all_emb_idx = torch.cat([all_emb_idx, batch_idx])
        
        batch_time[batch_time == 0] = float('inf')
        sorted_batch_time, indices_time = torch.sort(batch_time, dim=1)
        indices_time = indices_time.unsqueeze(-1).expand(-1, -1, self.input_size)
        batch = torch.gather(batch, 1, indices_time)
        all_emb_idx = torch.gather(
           all_emb_idx, dim=1, index=indices_time).float()
        sorted_batch_time[sorted_batch_time == float('inf')] = 0

        return batch, sorted_batch_time, all_emb_idx
    
class MultiModalTransformerSeqEncoder(M3SortTimeSeqEncoderContainerSameMod):
    def __init__(self,
                 trx_encoders,
                 input_size,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,
                 col_time='event_time',
                 **seq_encoder_params
                ):
        super().__init__(trx_encoders=trx_encoders,
                 seq_encoder_cls=XTransformerEncoder, 
                 input_size=input_size,
                 is_reduce_sequence=is_reduce_sequence,
                 col_time=col_time,
                 **seq_encoder_params)
        self.use_mask_of_padded = use_mask_of_padded
        self.pass_time_to_encoder = pass_time_to_encoder  

    def post_process_embs(self, x, length):
        return x, length
    
    def forward(self, x):
        #the modification is that time and mask are passed to the encoder
        #also there is a possibility of postprocess
        x, length = self.multimodal_trx_encoder(x)        
        x, length = self.post_process_embs(x, length)     
        x, merged_time, segm = self.merge_by_time(x)
        
        if self.use_mask_of_padded:
            mask  = torch.bitwise_and(merged_time > 0, merged_time != float('inf'))
        else:
            mask = None
            
        padded_x = PaddedBatch(payload=x, length=length)
        x = self.seq_encoder(padded_x, mask=mask, 
            time=merged_time if self.pass_time_to_encoder else None,
            segm=segm)
        return x   
        
class M3ConcatTimeSeqEncoderContainerSameMod(MultiModalTransformerSeqEncoder):
    def __init__(self,
                 trx_encoders,
                 input_size,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,
                 is_reduce_sequence=True,
                 col_time='event_time',
                 **seq_encoder_params
                ):
        super().__init__(trx_encoders=trx_encoders,
                 input_size=input_size,
                 is_reduce_sequence=is_reduce_sequence,
                 use_mask_of_padded=use_mask_of_padded,
                 pass_time_to_encoder=pass_time_to_encoder,
                 col_time=col_time,
                 **seq_encoder_params)
                
        self.token_sep = NoisyEmbedding(1, input_size, padding_idx=None,
                noise_scale=0.003)       
    
    def merge_by_time(self, x):
        #only concat and add sep_token in between, without sorting
        device = list(x.values())[1][0].device
        batch, batch_time = torch.tensor([], device=device), \
            torch.tensor([], device=device)
        all_emb_idx = torch.tensor([], device=device)

        batch_size = 0
        for i, source_batch in enumerate(x.values()):
            if source_batch[0] != 'None':
                batch_size = source_batch[1].payload.shape[0]
                break
                
        sep_idx = torch.zeros((batch_size, 1), dtype=int, device=device)
        
        for i, source_batch in enumerate(x.values()):
            if source_batch[0] != 'None':
                sep = self.token_sep(sep_idx)
                batch_idx = torch.ones(
                    (batch_size, source_batch[1].payload.shape[1]), dtype=int, device=device) \
                    * i
                    
                batch_to_cat = (batch, sep, source_batch[1].payload) \
                    if len(batch) > 0 else (batch, source_batch[1].payload)
                batch_time_to_cat = \
                    (batch_time, torch.ones_like(source_batch[0][:, 0].unsqueeze(1)), source_batch[0]) \
                    if len(batch) > 0 else (batch_time, source_batch[0])
                idx_to_cat = \
                    [all_emb_idx, torch.ones_like(source_batch[0][:, 0].unsqueeze(1)) * 0.5, batch_idx] \
                    if len(batch) > 0 else [all_emb_idx, batch_idx]
                
                batch = torch.cat(batch_to_cat, dim=1)
                batch_time = torch.cat(batch_time_to_cat, dim=1)
                all_emb_idx = torch.cat(idx_to_cat, dim=1)
        """
        batch_time[batch_time == 0] = float('inf')
        sorted_batch_time, indices_time = torch.sort(batch_time, dim=1)
        indices_time = indices_time.unsqueeze(-1).expand(-1, -1, self.input_size)
        batch = torch.gather(batch, 1, indices_time)
        sorted_batch_time[sorted_batch_time == 0] = float('inf')
        """
        return batch, batch_time, all_emb_idx    


class M3FusionModEncoderContainer(MultiModalTransformerSeqEncoder):
    def __init__(self,
                 trx_encoders,
                 input_size,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,
                 col_time='event_time',
                 concat=False,
                 mod_emb_dim=10,
                 **seq_encoder_params
                ):
        if concat:
            input_size += mod_emb_dim
            
        super().__init__(trx_encoders=trx_encoders,
                 input_size=input_size,
                 is_reduce_sequence=is_reduce_sequence,
                 use_mask_of_padded=use_mask_of_padded,
                 pass_time_to_encoder=pass_time_to_encoder,
                 col_time=col_time,
                 **seq_encoder_params)         
        
        mod_emb_size = mod_emb_dim if concat else input_size
        self.concat = concat
        self.mod_to_num = {mod: num for (num, mod) in enumerate(trx_encoders.keys())}
        self.mods_embs = NoisyEmbedding(len(self.mod_to_num), mod_emb_size, padding_idx=None,
                noise_scale=0.000)
        
    def merge_by_time(self, x):
        device = list(x.values())[1][0].device
        batch, batch_time = torch.tensor([], device=device), torch.tensor([], device=device)
        all_emb_idx = torch.tensor([], device=device)
        #add mod embeddings to source 
        for mod_name, source_batch in x.items():
            if source_batch[0] != 'None':             
                batch_size = source_batch[1].payload.shape[0]                
                seq_len = source_batch[1].payload.shape[1]
                emb_idx = torch.ones((batch_size, seq_len), dtype=int, device=device) * self.mod_to_num[mod_name]
                
                if self.concat:
                    concated = torch.cat(
                        (source_batch[1].payload, self.mods_embs(emb_idx)),
                        dim=2)
                    batch = torch.cat((batch, concated), dim=1)
                else:                
                    batch = torch.cat(
                        (batch, source_batch[1].payload + self.mods_embs(emb_idx)), 
                        dim=1)
                    
                batch_time = torch.cat((batch_time, source_batch[0]), dim=1)
                all_emb_idx = torch.cat((all_emb_idx, emb_idx), dim=1)

        batch_time[batch_time == 0] = float('inf')
        sorted_batch_time, indices_time = torch.sort(batch_time, dim=1)

        all_emb_idx = torch.gather(
           all_emb_idx.detach(), dim=1, index=indices_time).float()
        #print(all_emb_idx)
        batch = get_tensor_by_indices(batch, indices_time)
        sorted_batch_time[sorted_batch_time == float('inf')] = 0
        return batch, sorted_batch_time, all_emb_idx
    
import ciso8601
import time

class M3FusionResampleEncoderContainer(M3FusionModEncoderContainer):
    def __init__(self,
                 trx_encoders,
                 input_size,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,
                 col_time='event_time',
                 concat=False,
                 mod_emb_dim=10,
                 **seq_encoder_params
                ):            
        super().__init__(trx_encoders=trx_encoders,
                 input_size=input_size,
                 is_reduce_sequence=is_reduce_sequence,
                 use_mask_of_padded=use_mask_of_padded,
                 pass_time_to_encoder=pass_time_to_encoder,
                 col_time=col_time,
                 concat=concat,
                 mod_emb_dim=mod_emb_dim,      
                 **seq_encoder_params)         
        
        self.zero_token = NoisyEmbedding(1, input_size, padding_idx=None, noise_scale=0.003)
        self.pad_token = NoisyEmbedding(1, input_size, padding_idx=None, noise_scale=0.003)
    
    def post_process_embs(self, x, length):
        for mod_name in x.keys():
            mod_batch_time, mod_batch_feats = x[mod_name]
            min_time = "2016-04-21 00:00:00"
            ts = ciso8601.parse_datetime(min_time)
            # to get time in seconds:
            min_time = time.mktime(ts.timetuple())
            #print(ts.timetuple())
            max_time = "2016-06-22 00:00:00"
            ts = ciso8601.parse_datetime(max_time)
            # to get time in seconds:
            max_time = time.mktime(ts.timetuple())
            ints_num = 256
            interval = int(math.ceil((max_time - min_time) / ints_num))
            
            mod_batch_time, mod_batch_feats = resample(mod_batch_feats.payload, mod_batch_time, 
                    interval, normalize=True, wtype='sum', 
                    zero_token=self.zero_token,
                    pad_token=self.pad_token,
                    max_len=ints_num, sigma=0.5, min_time=int(min_time), 
                                                       cut_after_end=True)
            
            mod_batch_feats = PaddedBatch(payload=mod_batch_feats, length=mod_batch_feats.shape[1])
            x[mod_name] =  mod_batch_time, mod_batch_feats
            
        return x, length
    
    
class M3FusionAddTokenContainer(M3FusionModEncoderContainer):
    def __init__(self,
                 trx_encoders,
                 input_size,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,
                 col_time='event_time',
                 concat=False,
                 mod_emb_dim=10,
                 **seq_encoder_params
                ):            
        super().__init__(trx_encoders=trx_encoders,
                 input_size=input_size,
                 is_reduce_sequence=is_reduce_sequence,
                 use_mask_of_padded=use_mask_of_padded,
                 pass_time_to_encoder=pass_time_to_encoder,
                 col_time=col_time,
                 concat=concat,
                 mod_emb_dim=mod_emb_dim,      
                 **seq_encoder_params)         
        
        self.space_tokens = torch.nn.Parameter(torch.randn(1, 1, input_size),# / math.sqrt(input_size), 
                                             requires_grad=True)

    
    def post_process_embs(self, x, length):
        for mod_name in x.keys():
            mod_batch_time, mod_batch_feats = x[mod_name]
            interval = 24 * 60 * 60#int(math.ceil((max_time - min_time) / ints_num))
            #print(mod_batch_feats.payload.shape[1])
            mod_batch_time, mod_batch_feats = add_space_and_session_tokens(mod_batch_feats.payload, mod_batch_time, 
                    interval, 0,space_token=self.space_tokens, session_token=None)
            mod_batch_feats = PaddedBatch(payload=mod_batch_feats, length=mod_batch_feats.shape[1])
            #print(mod_batch_feats.payload.shape[1])
            x[mod_name] =  mod_batch_time, mod_batch_feats
            
        return x, length
            
    
class M3FusionInvertedEncoderContainer(MultiModalTransformerSeqEncoder):
    def __init__(self,
                 trx_encoders,
                 seq_encoder_cls, 
                 input_size,
                 is_reduce_sequence=True,
                 col_time='event_time',
                 concat=False,
                 mod_emb_dim=10,
                 **seq_encoder_params
                ):            
        super().__init__(trx_encoders=trx_encoders,
                 seq_encoder_cls=seq_encoder_cls, 
                 input_size=input_size,
                 is_reduce_sequence=is_reduce_sequence,
                 col_time=col_time,
                 concat=concat,
                 mod_emb_dim=mod_emb_dim,      
                 **seq_encoder_params)         
    
    def forward(self, x):
        x, length = self.multimodal_trx_encoder(x)
        
        for mod_name in x.keys():
            mod_batch_time, mod_batch_feats = x[mod_name]
            mod_batch_time, mod_batch_feats = resample(mod_batch_feats.payload, mod_batch_time, 
                    2 * 60 * 60, normalize=False, wtype='sum', zero_token=self.zero_token, 
                                                       max_len=256, sigma=0.5)#self.zero_token)
            #8 * 60 * 60, normalize=False, wtype='kde', sigma=60.0)
            mod_batch_feats = PaddedBatch(payload=mod_batch_feats, length=mod_batch_feats.shape[1])
            x[mod_name] =  mod_batch_time, mod_batch_feats
            
        x = self.merge_by_time(x)
        padded_x = PaddedBatch(payload=x, length=x.shape[1])
        x = self.seq_encoder(padded_x)
        return x
    
    """
    def merge_by_time(self, x):
        device = list(x.values())[1][0].device
        batch, batch_time = torch.tensor([], device=device), torch.tensor([], device=device)
        
        #add mod embeddings to source 
        for mod_name, source_batch in x.items():
            if source_batch[0] != 'None':  
                batch_size = source_batch[1].payload.shape[0]
                seq_len = source_batch[1].payload.shape[1]
                
                if self.concat:
                    concated = torch.cat((source_batch[1].payload, 
                                       self.mods_embs[mod_name].expand(batch_size, seq_len, -1)), dim=2)
                    batch = torch.cat((batch, concated), dim=1)
                else:                
                    batch = torch.cat((batch, source_batch[1].payload + 
                                   self.mods_embs[mod_name].expand(batch_size, seq_len, -1)), dim=1)
                    
                batch_time = torch.cat((batch_time, source_batch[0]), dim=1)
        
        batch_time[batch_time == 0] = float('inf')
        indices_time = torch.argsort(batch_time, dim=1)
        batch = self.get_tensor_by_indices(batch, indices_time)
        #print(indices_time.shape)
        #print(batch_time.shape)
        batch_time[:, indices_time][torch.arange(batch_size), torch.arange(batch_size), :]
        #batch_time = batch_time[:,indices_time]
        #print(batch_time.shape)
        batch = resample(batch, batch_time.cpu().numpy(), 3 * 60 * 60)
        return batch
    """
    

class EarlyFusionMatchingModule(WarmupMixin, ABSModule):
    def __init__(self,
                 fusion='middle',
                 use_diffs=True,
                 use_time_hist=False,
                 multimodal_seq_encoder=None,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                warmup_steps = 500,
                 initial_lr = 0.001):        
        
        if loss is None:
            loss = torch.nn.BCEWithLogitsLoss()
            
        if head is None:
            input_size = multimodal_seq_encoder.embedding_size
            if fusion == 'middle' and use_diffs:
                input_size *= 2
                
            if use_time_hist:
                input_size += 4 * 2# + 24 * 2 + 24 * 7 * 2
            
            head = Head(input_size=input_size, use_batch_norm=False, 
                    hidden_layers_sizes=[512, 512, 512, 512], objective="regression", 
                    num_classes=1)
            
        super().__init__(validation_metric=validation_metric,
                         seq_encoder=multimodal_seq_encoder,
                         loss=loss,
                         optimizer_partial=optimizer_partial,
                         lr_scheduler_partial=lr_scheduler_partial,
                         warmup_steps=warmup_steps,
                        initial_lr=initial_lr)
        self._head = head
        self.sigmoid = torch.nn.Sigmoid()
        self.y_cache = []
        self.sim_cache = []
        self.losses = defaultdict(list)
        self.fusion = fusion
        self.use_diffs = use_diffs
        self.use_time_hist = use_time_hist
        
    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True
    
    def embs_middle_fusion(self, x):
        if self.use_time_hist:
            seq_x, non_seq_x = x
            x = seq_x
            
        multitrx, l = self._seq_encoder.multimodal_trx_encoder(x)
        embs = []
        
        for mod, (mod_time, mod_trx) in multitrx.items():          
            curr_emb = self._seq_encoder.seq_encoder(mod_trx)
            embs.append((mod, curr_emb))
        
        ems_prod = torch.ones_like(embs[0][1])
        
        for i in range(len(embs)):
            ems_prod *= embs[i][1]
            
        embs_cat = ems_prod
        if self.use_diffs:
            ems_diff = torch.ones_like(embs[0][1]) * embs[0][1]
            for i in range(1, len(embs)):
                ems_diff -= embs[i][1]
            embs_cat =torch.cat((embs_cat, ems_diff), dim=1)
            
        if self.use_time_hist:
            #print(list(non_seq_x.keys()))
            non_seq_x = list(non_seq_x.values())
            h1, h2 = non_seq_x
            res = []
            keys = h1[0].keys()
            
            for i in range(len(h1)):
                curr_res = []
                for k in keys:
                    """
                    for j in range(len(h1[i][k])):
                        curr_res.append(h1[i][k][j] * h2[i][k][j])

                    for j in range(len(h1[i][k])):
                        curr_res.append(h1[i][k][j] - h2[i][k][j])
                    """
                    curr_res.append(pearson_correlation(h1[i][k],h2[i][k]))
                    curr_res.append(cosine_distance(h1[i][k],h2[i][k]))
                    curr_res.append(pearson_correlation(np.log(h1[i][k]+1.0),np.log(h2[i][k]+1.0)))
                    curr_res.append(cosine_distance(np.log(h1[i][k]+1.0),np.log(h2[i][k]+1.0)))
                res.append(np.array(curr_res))
                
            #print(res)
            res = np.array(res).astype(np.float32)
            
            res = torch.from_numpy(np.array(res)).to('cuda')
            embs_cat =torch.cat((embs_cat, res), dim=1)
        
        """
        ems_prod = torch.tensor([], device='cuda')
        inds = np.random.permutation(len(embs))
        for i in inds:
            ems_prod = torch.cat((ems_prod, embs[i][1]), dim=1)
        """
            
        return embs_cat
    
    def embs_early_fusion(self, x):  
        return self._seq_encoder(x)  
    
    def forward(self, x, calc_proba=False): 
        #print(x['trx'].__dict__)  
        if self.fusion == 'early':
            embs = self.embs_early_fusion(x)
        elif self.fusion == 'middle':
            embs = self.embs_middle_fusion(x)
        else:
            raise NotImplementedError(self.fusion)
        
        sim = self._head(embs)    
        if (not self.training) and calc_proba:
            sim = self.sigmoid(sim)
        sim = torch.squeeze(sim)
        return sim
    
    def shared_step(self, x, y):
        sim = self(x)        
        y = torch.as_tensor(np.array(y, dtype=np.float32)).to('cuda')
        sim = sim.float()
        return sim, y
        
    def training_step(self, batch, _):
        x, users, y = batch
        torch.set_float32_matmul_precision("high")
        sim, y = self.shared_step(x, y)
        loss = self._loss(sim, y)
        self.losses['train'].append(loss.detach().cpu())

        #print(batch)
        if self.use_time_hist:
            seq_x, non_seq_x = x
            x = seq_x
            
        for mod_name, mod_batch in x.items():
            self.log('seq_len/' + mod_name, x[mod_name].seq_lens.float().mean(), prog_bar=True)    
        return loss
    
    def on_train_epoch_end(self):
        self.log_loss('train')

    def log_loss(self, stage):
        curr_loss = np.array(self.losses[stage]).mean()
        self.log(stage + r'/loss', curr_loss)
        self.losses[stage] = []

    def validation_step(self, batch, _):
        x, users, y = batch
        sim, y = self.shared_step(x, y)
        loss = self._loss(sim, y)
        self.losses['val'].append(loss.detach().cpu())

        y = y.type('torch.LongTensor').to('cuda')
        self.y_cache.append(y.detach().cpu().numpy().ravel())
        self.sim_cache.append(sim.detach().cpu().float().numpy().ravel())
    
    def on_validation_epoch_end(self):
        self.log_loss('val')
        self.y_cache = np.hstack(self.y_cache)
        self.sim_cache = np.hstack(self.sim_cache)
        
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(self.y_cache, self.sim_cache)
        f1_scores = 2*recall*precision/(recall+precision + 0.000000001)
        best_ind = np.argmax(f1_scores)        
        roc_auc = roc_auc_score(self.y_cache, self.sim_cache)
        
        self.y_cache = []
        self.sim_cache = []
        self.log('valid/recall', recall[best_ind], prog_bar=True) 
        self.log('valid/prec', precision[best_ind], prog_bar=True) 
        self.log('valid/f1', f1_scores[best_ind], prog_bar=True) 
        self.log('valid/thresh', thresholds[best_ind], prog_bar=True) 
        self.log('valid/roc_auc', roc_auc, prog_bar=True) 
        
    