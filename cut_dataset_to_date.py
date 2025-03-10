def cut_dataset_to_date(parquet_in, low_date_thresh, high_date_thresh, mod_name, col_time, parquet_out):

import numpy as np
import pandas as pd
from tqdm import tqdm
from ptls.data_load.datasets.parquet_dataset import ParquetFiles
import os

class CutDatasetToDate:
    def __init__(self, parquet_in, low_date_thresh, high_date_thresh, mod_names, col_time, parquet_out):
        if type(parquet_in) is str:
            parquet_in = [parquet_in]
         
        self.parquet_out = parquet_out
        self.parquet_in = parquet_in       
        self.mod_names = mod_names
        self.col_time = col_time
        self.low_date_thresh = low_date_thresh  
        self.high_date_thresh = high_date_thresh
        
    def read_data(self):
        print("read_data")        
        self.data_in = []
        parquet_helper = ParquetFiles(self.parquet_in)
        
        for f_name in tqdm(parquet_helper.data_files):
            print(f_name)
            curr_data = pd.read_parquet(f_name)
            splited_data = split_mod_data(curr_data)
            
            result_data = {}
            for mod in splited_data.keys():
                curr_data = splited_data[mod]
                result_data.update(cut_dataset_to_date(curr_data, mod))
            
            for col in curr_data.columns:
                if not(col in result_data):
                    result_data[col] = curr_data[col]
                    
            result_data = pd.DataFrame(data=result_data, index=curr_data.index.values)
            print(result_data.shape)
            print(list(result_data.columns))
            self.write_data(result_data, os.path.split(f_name)[1])
            
    def cut_dataset_to_date(self, data, mod_name):
        data_dict = {}
        for col in data.columns:
            data_dict[col] = data[col]
         
        times = data_dict[mod_name + "_" + self.col_time].values
        times_idx = []
        
        for elem in times:
            times_idx.append(np.bitwise_and(np.array(elem) >= self.low_date_thresh, 
                                   np.array(elem) >= self.high_date_thresh))
        for col in data_dict.keys():
            if not(type(data_dict[col].values[0]) is list):
                continue
            
            curr_arr = data_dict[col]
            new_arr = []
            for i in range(len(times_idx)):
                curr_idx = times_idx[i]
                new_arr.append(list(curr_arr[i][curr_idx]))
                
            data_dict[col] = np.array(new_arr)
            
        return data_dict
            
    def split_mod_data(self):
        splited_data = {}
        
        #print("modality_split", {k:v for k,v in feature_arrays.items()})
        for mod_name in self.mod_names:
            curr_mod_cols = [k for k in curr_data.columns if k.startswith(mod_name)]
            splited_data[mod_name] = curr_data[curr_mod_cols]
            
        return splited_data
            
    def write_data(self, curr_data, fname):
        print("write_pairs")        
        
        if not os.path.exists(self.parquet_out):
            os.makedirs(self.parquet_out)

        fout = os.path.join(self.parquet_out, fname)                
        curr_data.to_parquet(fout)