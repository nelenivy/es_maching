from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
import numpy as np
import torch
import ciso8601
import time
from ptls.data_load.feature_dict import FeatureDict

def str2milisec(time_str):
    return time.mktime(ciso8601.parse_datetime(time_str).timetuple())

class LimitTime(IterableProcessingDataset):    
    def __init__(self, mod_names, col_time, high_date_thresh=None, low_date_thresh=None):
        super().__init__()
        self.mod_names = mod_names
        self.col_time = col_time   
        self.high_date_thresh = high_date_thresh

        if self.high_date_thresh:
            self.high_date_thresh = str2milisec(self.high_date_thresh) 

        self.low_date_thresh = low_date_thresh
        
        if self.low_date_thresh:
            self.low_date_thresh = str2milisec(self.low_date_thresh)
            print(self.low_date_thresh)
        
    def __iter__(self):
        for rec in self._src:
            #print(1)
            features = rec[0] if type(rec) is tuple else rec
            is_empty = False
            
            for mod in self.mod_names:
                seq_time = np.array(features[mod + "_" + self.col_time])
                curr_idx = np.ones(seq_time.shape, dtype=bool)
                if self.high_date_thresh:
                    curr_idx = np.bitwise_and(curr_idx, seq_time <= self.high_date_thresh)
                
                if self.low_date_thresh:
                    curr_idx = np.bitwise_and(curr_idx, seq_time >= self.low_date_thresh)
                curr_idx = curr_idx.astype(bool)
                #print("---------------------------------------------------")
                #print(curr_idx)
                #print(mod, curr_idx.dtype, curr_idx.sum(), curr_idx.shape)
                if curr_idx.sum() == 0:
                    is_empty = True
                    break
                for feature in features.keys():
                    if feature.startswith(mod):
                        #print(mod, feature)
                        curr_feat = features[feature]
                        if FeatureDict.is_seq_feature(feature, curr_feat) or (type(curr_feat) is list):
                            #print(feature, features[feature].shape)
                            features[feature] = np.array(curr_feat)[curr_idx]
                            #print(curr_feat, features[feature].shape)
                            
            if is_empty:
                continue
            else:    
                yield rec
