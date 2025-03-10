import numpy as np
import pandas as pd

def sort_pair(pair):
    return (min(pair[0],pair[1]),max(pair[0],pair[1]))

def sort_pair_with_dist(pair_with_dist):
    return (*sort_pair(pair_with_dist[:2]), pair_with_dist[2])

def make_data(data, col1, col2):
    data1 = data.loc[np.array(col1), :]
    data2 = data.loc[np.array(col2), :]
    data1 = data1.rename(columns={"data_time": "data1_time", "data_url0": "data1_url0", 
                          "data_url1": "data1_url1",
                         "data_url2": "data1_url2", 
                          "data_url3": "data1_url3", 
                                  'uid': 'data1_uid'})
    data2 = data2.rename(columns={"data_time": "data2_time", "data_url0": "data2_url0", 
                          "data_url1": "data2_url1",
                         "data_url2": "data2_url2", 
                          "data_url3": "data2_url3", 
                                  'uid': 'data2_uid'})
    #test_data = data.loc[np.array(users_for_predict), :]
    return pd.concat([data1.reset_index(drop=True, inplace=False), data2.reset_index(drop=True, inplace=False)], axis=1)

def make_data_pairs(data, pairs):
    col1 = []
    col2 = []
    for c1, c2 in pairs:
        col1.append(c1)
        col2.append(c2)
        
    return make_data(data, col1, col2)