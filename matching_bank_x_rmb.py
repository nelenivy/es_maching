import torch
import numpy as np
from collections import defaultdict

from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset

from functools import reduce
from operator import add

from ptls.data_load.feature_dict import FeatureDict
from ptls.frames.coles.split_strategy import AbsSplit
from ptls.frames.coles.metric import metric_recall_top_K, outer_cosine_similarity, outer_pairwise_distance
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from warmup_mixin import WarmupMixin
from m3_utils import batch_nearest_neighbors

def first(iterable, default=None):
    iterator = iter(iterable)
    return next(iterator, default)

def split_dict_of_lists_by_one(dict_of_lists):
    list_len = len(first(dict_of_lists.values()))
    to_return = [{} for _ in range(list_len)]
    
    for k, elems_list in dict_of_lists.items():
        for ind, elem in enumerate(elems_list):
            to_return[ind][k] = [elem]
            
    return to_return
    
class AmtProc(IterableProcessingDataset):
    def __init__(self, amt_cols):
        super().__init__()
        if amt_cols is None:
            amt_cols = [] 
        self._amt_cols = amt_cols
        
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            
            for col in self._amt_cols:
                features[col] = np.nan_to_num(np.array(features[col]).astype('float32').view('float32'))
            yield rec

class PrepareOnePairMixin:
    def _prepare_one_pair(self, feature_arrays):
        # print()
        # print("_prepare_one_pair.feature_arrays:", feature_arrays)
        # print()
        modalities = self.modality_split(feature_arrays)
        # print()
        # print("_prepare_one_pair.modalities:", modalities)
        # print()
        return self.get_splits(modalities)
    
class PrepareOnePairSupervisedMixin:
     def _prepare_one_pair(self, feature_arrays):
        modalities = self.modality_split(feature_arrays)   
        splited_modalities = self.get_splits(modalities)
        splits_num = len(first(splited_modalities.values()))
        classes_for_splits = self.get_classes(feature_arrays, splits_num)
        return splited_modalities, classes_for_splits
    
class M3ColesDatasetBase(FeatureDict, torch.utils.data.Dataset):
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
                 mod_names = ['bank', 'rmb'],
                 yield_by_one=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time
        self.mod_names = mod_names
        self.yield_by_one = yield_by_one

    def __len__(self):
        return len(self.data)    
    

    def __getitem__(self, idx):
        return self._prepare_one_pair(self.data[idx])
        
    def __iter__(self):
        for feature_arrays in self.data:
            # print()
            #print("__iter__ feature_arrays", [type(t) for t in feature_arrays.values()])    
            #CHECK IF THERE IS AN EMPTY ARRAY OF FEATURES, IF IT IS THEN CONTINUE
            is_empty = False
            for feat_name, feat in feature_arrays.items():
                if isinstance(feat, torch.Tensor):
                    #print(feat.shape)
                    if feat.shape[0] == 0:
                        is_empty = True
                        break
            
            if is_empty:
                continue
                
            splited_pairs = self._prepare_one_pair(feature_arrays)
            
            if self.yield_by_one:
                splited_pairs = self.split_one_by_one(splited_pairs)
                for one_pair in splited_pairs:
                    yield one_pair
            else:
                yield splited_pairs

    def split_one_by_one(self, _):
        raise NotImplementedError("")
        
    @staticmethod
    def apply_splits_to_feat_dict(feature_arrays, indexes):
        return [{k: v[ix] if M3ColesDataset.is_seq_feature(k, v) else v for k, v in feature_arrays.items() } for ix in indexes]
    
    def get_splits(self, modalities):
        return_splits = {}
        #print("get_splits.modalities.items: ", modalities.items())
        # print()
        for mod_name, feature_arrays in modalities.items():
            return_splits[mod_name] = self.get_one_modality_time_split(feature_arrays)
        return return_splits

    def get_one_modality_time_split(self, feature_arrays):
        #print()
        #print("get_one_modality_time_split.features_arrays", feature_arrays)
        #print()
        #print(feature_arrays)
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        return self.apply_splits_to_feat_dict(feature_arrays, indexes)
    
    def modality_split(self, feature_arrays):
        modalities = {}
        #print()
        #print("modality_split.feature_arrays.items()", feature_arrays.items())
        #print(self.mod_names)
        # print()
        for mod_name in self.mod_names:
            modalities[mod_name] = {k.replace(mod_name + '_', ''):v for k,v in feature_arrays.items() if k.startswith(mod_name)}
        return modalities

    @staticmethod
    def collate_fn(batch):
        class_labels = []
        
        for i, class_samples in enumerate(batch):
            one_mod_split_samples = first(class_samples.values()) 
            class_labels += [i] * len(one_mod_split_samples)
            
        mod_names = list(batch[0].keys())
        #print("batch_labels")
        #print(len(batch))
        padded_batches = {}
        for mod in mod_names:
            reduced_mod = reduce(add, [b[mod] for b in batch])
            #print(len(reduced_mod))
            padded_batches[mod] = collate_feature_dict(reduced_mod)
        #print("collate ")
        return padded_batches, torch.LongTensor(class_labels)

class M3ColesDataset(M3ColesDatasetBase, PrepareOnePairMixin):
    pass

class M3ColesIterableDataset(M3ColesDataset, torch.utils.data.IterableDataset):
    pass


class M3ColesSupervisedDatasetMixin(PrepareOnePairSupervisedMixin):
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
                 cols_classes = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class

        self.cols_classes = cols_classes
        
    
    def get_classes(self, feature_arrays, split_count):
        classes = [feature_arrays.get(col, -1) for col in self.cols_classes]
        res = []
        
        for l in classes:
            res += [l for _ in range(split_count)]
            
        return res
    
    
    @staticmethod
    def collate_fn(batch):
        feats_batch = [b[0] for b in batch]
        #print("feats_batch")
        collated_feats, split_labels = M3ColesDataset.collate_fn(feats_batch)
        #print("feats_collate_fn")
        target_labels = []
        for seq, labels in batch:
            target_labels += labels
        
        return collated_feats, split_labels, target_labels

class M3ColesSupervisedDataset(M3ColesDatasetBase, M3ColesSupervisedDatasetMixin):
    def __init__(self,
                 data,
                 splitter: AbsSplit,
                 col_time='event_time',
                 mod_names = ['bank', 'rmb'],
                 cols_classes = None,
                 *args, **kwargs):
        super().__init__(data=data, splitter=splitter,
                 col_time=col_time,
                 mod_names=mod_names, cols_classes=cols_classes, 
                         *args, **kwargs)  # required for mixin class
        
    @staticmethod
    def collate_fn(batch):
        return M3ColesSupervisedDatasetMixin.collate_fn(batch)

class M3ColesSupervisedIterableDataset(M3ColesSupervisedDataset, torch.utils.data.IterableDataset):
    pass


def metric_recall_top_K_for_embs(embs_1, embs_2, true_matches, K=100):
    nearest_indices = batch_nearest_neighbors(embs_1, embs_2, batch_size=1000, K=K, return_dists=False)
    embs = [embs_1, embs_2]
    # file=open('err.txt','a+')
    recall_at_k = np.zeros(len(nearest_indices))

    for mod_ind in (0, 1):
        correct_matches = 0
        print(len(embs_1), nearest_indices[mod_ind].shape[0])
        for i, indices in enumerate(nearest_indices[mod_ind]):
            if true_matches[i].numpy() in list(indices):
                correct_matches += 1
            # print(true_matches[i], file=file)
            # print(indiёces, type(indices), file=file)
            if i < 10:
                print(true_matches[i].numpy(), indices)
        recall_at_k[mod_ind] = correct_matches / len(nearest_indices[mod_ind])

    # file=open('err.txt','a+')
    # print('\n', recall_at_k, file=file)
 
    return recall_at_k    
    
    
class M3CoLESModule(WarmupMixin, ABSModule):
    """
    Multi-Modal Matching
    Contrastive Learning for Event Sequences ([CoLES](https://arxiv.org/abs/2002.08232))

    Subsequences are sampled from original sequence.
    Samples from the same sequence are `positive` examples
    Samples from the different sequences are `negative` examples
    Embeddings for all samples are calculated.
    Paired distances between all embeddings are calculated.
    The loss function tends to make positive distances smaller and negative ones larger.

    Parameters
        seq_encoder:
            Model which calculate embeddings for original raw transaction sequences
            `seq_encoder` is trained by `CoLESModule` to get better representations of input sequences
        head:
            Model which helps to train. Not used during inference
            Can be normalisation layer which make embedding l2 length equals 1
            Can be MLP as `projection head` like in SymCLR framework.
        loss:
            loss object from `ptls.frames.coles.losses`.
            There are paired and triplet loss. They are required sampling strategy
            from `ptls.frames.coles.sampling_strategies`. Sampling strategy takes a relevant pairs or triplets from
            pairwise distance matrix.
        validation_metric:
            Keep None. `ptls.frames.coles.metric.BatchRecallTopK` used by default.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.

    """
    def __init__(self,
                 seq_encoders=None,
                 mod_names=None,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                warmup_steps = 500,
                 initial_lr = 0.001):
        torch.set_float32_matmul_precision('high')
        if head is None:
            head = Head(use_norm_encoder=True)

        if loss is None:
            loss = ContrastiveLoss(margin=0.5,
                                   sampling_strategy=HardNegativePairSelector(neg_count=5))

        if validation_metric is None:
            validation_metric = BatchRecallTopK(K=4, metric='cosine')
        
        for k in seq_encoders.keys():
            if type(seq_encoders[k]) is str:
                seq_encoders[k] = seq_encoders[seq_encoders[k]]
                
        super().__init__(validation_metric,
                         first(seq_encoders.values()),
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial,
                        warmup_steps=warmup_steps,
                 initial_lr=initial_lr)
        print("seq_encoders", seq_encoders)
        self.seq_encoders = torch.nn.ModuleDict(seq_encoders)
        self._head = head   
        self.y_h_cache = {'train':[], 'valid': []}
        self.valid_loss_cache = []
        
    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True
    
    def forward(self, x):
        res = {}
        #print(x['trx'].__dict__)
        for mod_name in x.keys():
            res[mod_name] = self.seq_encoders[mod_name](x[mod_name])
            
        return res

    def shared_step(self, x, y):
        y_h = self(x)
        
        if self._head is not None:
            y_h_head = {k: self._head(y_h_k) for k, y_h_k in y_h.items()}
            y_h = y_h_head
            
        return y_h, y
    
    def _one_step(self, batch, _, stage):
        #print(batch)
        y_h, y = self.shared_step(*batch)
        y_h_list = list(y_h.values())
        mod_num = len(y_h_list)
        mod_segms = [torch.ones_like(y) * i for i in range(mod_num)]

        loss = self._loss(torch.cat(y_h_list), 
                          torch.cat([y] * mod_num),
                         torch.cat(mod_segms))

        if stage == "train":
            self.log(f'loss/{stage}', loss.detach())
        
        x, y = batch

        for mod_name, mod_x in x.items():
            self.log(f'seq_len/{stage}/{mod_name}', x[mod_name].seq_lens.float().mean().detach(), prog_bar=True)
        
        if stage == "valid":
            self.valid_loss_cache.append(loss.detach().cpu().numpy())
            n, d = y_h_list[0].shape
            y_h_concat = torch.zeros((2*n, d), device = y_h_list[0].device)

            for i in range(2):
                y_h_concat[range(i,2*n,2)] = y_h_list[i] 

            if len(self.y_h_cache[stage]) <= 380:
                self.y_h_cache[stage].append((y_h_concat.cpu(), 
                    {k: y_h_k.cpu() for k, y_h_k in y_h.items()} , 
                    {k:x_k.seq_lens.cpu() for k, x_k in x.items()})) 
    
        return loss
    
    def training_step(self, batch, _):
        #print(batch, _)
        return self._one_step(batch, _, "train")
    
    def validation_step(self, batch, _):
        return self._one_step(batch, _, "valid")
    
    # jnkflbnmm
    def on_validation_epoch_end(self):   
        self.log('loss/valid', np.mean(self.valid_loss_cache))
        # print("METRICS", np.mean(self.valid_loss_cache), '\n', file)
        len_intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 80), (80, 120), (120, 160), (160, 240)]
        self.log_recall_top_K(self.y_h_cache['valid'], len_intervals=len_intervals, stage="valid", K=100)
        
        del self.y_h_cache["valid"]
        self.y_h_cache["valid"] = []
        
    def log_recall_top_K(self, y_h_cache, len_intervals=None, stage="valid", K=100):
        #print(y_h_cache)
        y_h = torch.cat([item[0] for item in y_h_cache], dim = 0)
        y_h_mods = defaultdict(list)
        seq_lens_dict = defaultdict(list)
        
        for item in y_h_cache:
            for k, emb in item[1].items():
                y_h_mods[k].append(emb)
            for k, l in item[2].items():
                seq_lens_dict[k].append(l)
        
        y_h_mods = {k: torch.cat(el, dim=0) for k ,el in y_h_mods.items()}
        seq_lens_dict = {k: torch.cat(el) for k ,el in seq_lens_dict.items()}

        y_h_bank, y_h_rmb = list(y_h_mods.values())
        computed_metric_b2r, computed_metric_r2b = metric_recall_top_K_for_embs(
            y_h_bank, y_h_rmb, torch.arange(y_h_rmb.shape[0]), K)
        #computed_metric_r2b = metric_recall_top_K_for_embs(
        #    y_h_rmb, y_h_bank, torch.arange(y_h_rmb.shape[0]), K)
        #file=open('err.txt','a+')
        print(computed_metric_r2b)#, file=file)
        print(computed_metric_b2r)#, file=file)

        if len_intervals != None:
            for mod, seq_lens in seq_lens_dict.items():
                for start, end in len_intervals:
                    mask = ((seq_lens > start) & (seq_lens <= end))

                    if torch.any(mask):
                        y_h_bank_filtered = y_h_bank[mask]
                        y_h_rmb_filtered = y_h_rmb[mask]


                        recall_r2b,recall_b2r  = metric_recall_top_K_for_embs(y_h_rmb_filtered, y_h_bank_filtered, torch.arange(y_h_rmb_filtered.shape[0]), K=K)
                        #recall_b2r = metric_recall_top_K_for_embs(y_h_bank_filtered, y_h_rmb_filtered, torch.arange(y_h_rmb_filtered.shape[0]), K=K)

                        #self.log(f"{mode}/R@100_len_from_{start}_to_{end}", recall, prog_bar=True)
                        self.log(f"{stage}/{mod}/r2b_R@100_len_from_{start}_to_{end}", recall_r2b, prog_bar=True)
                        self.log(f"{stage}/{mod}/b2r_R@100_len_from_{start}_to_{end}", recall_b2r, prog_bar=True)
                        #print(f"{stage}/{mod}/r2b_R@100_len_from_{start}_to_{end}", recall_r2b, file=file)
                        #print(f"{stage}/{mod}/b2r_R@100_len_from_{start}_to_{end}", recall_b2r, file=file)
        
        #self.log(f"{mode}/R@100", computed_metric, prog_bar=True)
        self.log(f"{stage}/r2b_R@100", computed_metric_r2b, prog_bar=True)
        self.log(f"{stage}/b2r_R@100", computed_metric_b2r, prog_bar=True)
        #print(f"{stage}/r2b_R@100", computed_metric_r2b, file=file)
        #print(f"{stage}/b2r_R@100", computed_metric_b2r, file=file)
        #print('\n', file=file)