import dask.dataframe as dd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pyarrow as pa
import numpy as np
import pandas as pd
from tqdm import tqdm
from matching_utils import sort_pair, make_data_pairs, make_data
from dataset_dists import DatasetEmbeddingsCalc

class PairsComposer(object):
    def __init__(self, embs, pos_pairs_f, neg_pairs_rate):
        self.pos_pairs_f = pos_pairs_f
        self.neg_pairs_rate = neg_pairs_rate
        self.df = embs        
            
    def nearest_neighbors(self):
        print("nearest_neighbors")
        self.N_NEIGHBOURS = 30

        knn = NearestNeighbors(n_neighbors=self.N_NEIGHBOURS, metric='cosine')
        knn = knn.fit(self.df.values)
        dists, self.neighb_inds = knn.kneighbors()
    
    def read_pos_pairs(self):
        print("read_pos_pairs")
        self.pos_col1 = []
        self.pos_col2 = []
        self.pos_pairs = set()
        
        with open(self.pos_pairs_f) as f_in:
            for line in tqdm(f_in):
                user1,user2 = line.strip().split(',')
                self.pos_col1.append(user1)
                self.pos_col2.append(user2)
                self.pos_pairs.add(sort_pair([user1,user2]))
        
    def neg_pairs(self):
        print("neg_pairs")
        self.neg_pairs = []
        already_seen = set()

        for i in range(self.df.shape[0]):
            curr_elem = self.df.index.values[i]
            for ind in range(self.N_NEIGHBOURS):
                curr_neighb_ind = self.neighb_inds[i, ind]                
                
                if curr_neighb_ind == i:
                    continue
                    
                curr_neighb = self.df.index.values[curr_neighb_ind]
                pair = sort_pair([curr_elem, curr_neighb])

                if not (pair in already_seen):
                    already_seen.add(pair)

                    if not (pair in self.pos_pairs):
                        self.neg_pairs.append(pair)
                        
    def combine_pos_and_neg(self):
        print("combine_pos_and_neg")
        neg_pairs_perm = np.random.permutation(self.neg_pairs)
        pos_pairs_num = len(self.pos_pairs)
        neg_pairs_num = min(int(self.neg_pairs_rate * pos_pairs_num), len(neg_pairs_perm))
        self.neg_pairs_to_take = neg_pairs_perm[:neg_pairs_num]        
    
    def process(self):
        self.nearest_neighbors()
        self.read_pos_pairs()
        self.neg_pairs()
        self.combine_pos_and_neg()
    
class MidFusionConverter:
    def __init__(self, parquet_in, pos_pairs_f, yaml_in, checkpoint, neg_pairs_rate, parquet_out):
        self.embeddings_computer = DatasetEmbeddingsCalc(parquet_in, yaml_in, checkpoint)
        self.embs = self.embeddings_computer.process()
        self.pairs_composer = PairsComposer(self.embs, pos_pairs_f, neg_pairs_rate)
        self.parquet_out = parquet_out
        self.parquet_in = parquet_in
        
    def read_data(self):
        print("read_data")
        self.data_in = dd.read_parquet(self.parquet_in).compute()
        if 'uid' in self.data_in.columns:
            self.data_in = self.data_in.set_index('uid', drop=False)
            
    def make_data(self):
        self.read_data()
        self.pairs_composer.process()
        neg_data = make_data_pairs(self.data_in, self.pairs_composer.neg_pairs_to_take)
        neg_data['label'] = np.zeros(neg_data.shape[0], dtype=int)
        pos_data = make_data(self.data_in, self.pairs_composer.pos_col1, self.pairs_composer.pos_col2)
        pos_data['label'] = np.ones(pos_data.shape[0], dtype=int)
        self.mid_fusion_data = pd.concat(
            (neg_data.reset_index(drop=True, inplace=False),
             pos_data.reset_index(drop=True, inplace=False)), axis=0).reset_index(drop=True, inplace=False)
        all_pairs_perm = self.mid_fusion_data.index.values[np.random.permutation(self.mid_fusion_data.shape[0])]
        self.mid_fusion_data = self.mid_fusion_data.loc[all_pairs_perm, :].reset_index(drop=True, inplace=False)
        
    def write_pairs(self):
        print("write_pairs")
        self.mid_fusion_data_dask = dd.from_pandas(self.mid_fusion_data, chunksize=50000)
        schema = pa.schema([
        ('label', pa.int64()),
        ('data1_time', pa.list_(pa.int64())),
        ('data1_url0', pa.list_(pa.int64())),
        ('data1_url1', pa.list_(pa.int64())),
        ('data1_url2', pa.list_(pa.int64())),
        ('data1_url3', pa.list_(pa.int64())),
        ('data1_uid', pa.string()),
        ('data2_time', pa.list_(pa.int64())),
        ('data2_url0', pa.list_(pa.int64())),
        ('data2_url1', pa.list_(pa.int64())),
        ('data2_url2', pa.list_(pa.int64())),
        ('data2_url3', pa.list_(pa.int64())),
        ('data2_uid', pa.string())
        ])
        self.mid_fusion_data_dask.to_parquet(self.parquet_out, schema=schema)
        
    def process(self):
        self.make_data()
        self.write_pairs()
        
