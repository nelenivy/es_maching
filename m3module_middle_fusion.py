import pyarrow.parquet as pq
import logging
import pandas as pd
import torch
import numpy as np
import math
from collections import defaultdict
import torchmetrics
import sklearn
from sklearn.metrics import roc_auc_score
from ptls.data_load.datasets import ParquetDataset, ParquetFiles
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load import IterableChain
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch

from functools import reduce
from operator import iadd
from typing import Union, List
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
from ptls.data_load.datasets import ParquetDataset
from ptls.data_load import read_pyarrow_file
from matching_bank_x_rmb import M3CoLESModule

logger = logging.getLogger(__name__)

class M3MidFusionModule(M3CoLESModule):
    def __init__(self,
                 use_diffs=True,
                 seq_encoders=None,
                 mod_names=None,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):
        if loss is None:
            loss = torch.nn.BCELossWithLogits()

        if validation_metric is None:
            validation_metric = torchmetrics.classification.Accuracy()
            
        self.use_diffs = use_diffs
        num_features_list = [seq_enc.embedding_size for seq_enc in seq_encoders.values() if not(type(seq_enc) is str)]
        num_features = num_features_list[0]   
        input_size = 2 * num_features if self.use_diffs else num_features
        
        if head is None:
            head = Head(input_size=input_size, use_batch_norm=False, 
                    hidden_layers_sizes=[512, 512, 16], drop_probs=[0.1, 0.1, 0.1], objective="regression", 
                    num_classes=1)
            
        super().__init__(seq_encoders=seq_encoders,
                 head=head,
                 loss=loss,
                 validation_metric=validation_metric,
                 optimizer_partial=optimizer_partial,
                 lr_scheduler_partial=lr_scheduler_partial)

        self.y_cache = []
        self.sim_cache = []              
        self.sigmoid = torch.nn.Sigmoid()
        
    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True
    
    def forward(self, x, use_sigmoid=False):
        y_h = super().forward(x)
        y_h_list = list(y_h.values())
        to_head = y_h_list[0] * y_h_list[1]
        if self.use_diffs:
            to_head = torch.concat((to_head, torch.abs(y_h_list[0] - y_h_list[1])), dim=1)
        sim = self._head(to_head)
        
        if not self.training and use_sigmoid:
            sim = self.sigmoid(sim)
        
        sim = torch.squeeze(sim)
        return sim

    def shared_step(self, x, nums, y):
        sim = self(x)        
        y = torch.as_tensor(np.array(y, dtype=np.float32)).to('cuda')
        return sim, y
        
    def _one_step(self, batch, _, stage):
        x, nums, y = batch
        sim, y = self.shared_step(x, nums, y)
        loss = self._loss(sim, y)
        self.log(f'loss/f{stage}', loss)            
        
        for mod_name, mod_x in x.items():
            self.log(f'seq_len/{stage}/{mod_name}', x[mod_name].seq_lens.float().mean(), prog_bar=True)
        
        if stage == 'valid':
            y = y.type('torch.LongTensor')
            self.y_cache.append(y.detach().cpu().numpy().ravel())
            self.sim_cache.append(sim.detach().cpu().numpy().ravel())
        
        return loss
    
    def training_step(self, batch, _):
        return self._one_step(batch, _, "train")
        
    def validation_step(self, batch, _):
        return self._one_step(batch, _, "valid")
        
    def on_validation_epoch_end(self):
        self.y_cache = np.hstack(self.y_cache)
        self.sim_cache = np.hstack(self.sim_cache)
        
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(self.y_cache, self.sim_cache)
        f1_scores = 2*recall*precision/(recall+precision + 0.000000001)
        best_ind = np.argmax(f1_scores)
        
        roc_auc = roc_auc_score(self.y_cache, self.sim_cache)
        
        del self.y_cache
        del self.sim_cache
        self.y_cache = []
        self.sim_cache = []
            
        self.log('valid/recall', recall[best_ind], prog_bar=True) 
        self.log('valid/prec', precision[best_ind], prog_bar=True) 
        self.log('valid/f1', f1_scores[best_ind], prog_bar=True) 
        self.log('valid/thresh', thresholds[best_ind], prog_bar=True) 
        self.log('valid/roc_auc', roc_auc, prog_bar=True) 
