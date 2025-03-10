from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
import numpy as np
from datetime import datetime
import torch

def timestamp2hr(timestamp):
    return datetime.fromtimestamp(timestamp).hour


def timestamp2hr_week(timestamp):
    time = datetime.fromtimestamp(timestamp)
    hr = time.hour
    wd = time.weekday()
    return wd*24+hr

def get_24hr_distribution(uid, u2f):
    count = defaultdict(float)
    for fact in u2f[uid]:
        count[timestamp2hr(fact[0])]+=1.0
    return np.array([count[_] for _ in range(24)])

            
class TimeProcMatching(IterableProcessingDataset):
    '''This class is used for generation weekday and hour features from used source'''
    def __init__(self, time_col, source):
        super().__init__()
        self._time_col = time_col
        self._source = source
        self.i = 0
        
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            #print(features)
            if self._time_col in features: 
                hours = []
                week_hours = []
                
                timestamp = np.array(features[self._time_col])
                timestamp[timestamp>9999999999] //= 1000
                timestamp[timestamp>9999999999] //= 1000
                
                for curr_time in timestamp:
                    hours.append(timestamp2hr(curr_time))
                    week_hours.append(timestamp2hr_week(curr_time))
                    
                features[self._source + '_hour'] = torch.from_numpy(np.array(hours))
                features[self._source + '_weekday'] = torch.from_numpy(np.array(week_hours))                       
                    
            yield rec
            
def convert_to_datetime_seconds(time_arr):
    timestamp = np.array(time_arr)
    timestamp[timestamp>9999999999] //= 1000
    timestamp[timestamp>9999999999] //= 1000
    ts = np.array(timestamp).astype(int).astype('datetime64[s]')
    return ts
    
class TimeProcMatchingFast(IterableProcessingDataset):
    '''This class is used for generation weekday and hour features from used source'''
    def __init__(self, time_col, source):
        super().__init__()
        self._time_col = time_col
        self._source = source
        
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            if self._time_col in features:
                ts = convert_to_datetime_seconds(features[self._time_col])
                day = np.array(ts).astype('datetime64[D]')
                features[self._source + '_weekday'] = (day.view('int64') - 4) % 7 + 1
                features[self._source + '_hour'] = (ts - day).astype('timedelta64[h]').view('int64')
                features[self._source + '_weekday'] = (features[self._source + '_weekday'] - 1)* 24 + features[self._source + '_hour']
            yield rec
            
class TimeProcMatchingFull(IterableProcessingDataset):
    '''This class is used for generation weekday and hour features from used source'''
    def __init__(self, time_col, source):
        super().__init__()
        self._time_col = time_col
        self._source = source
        
    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            if self._time_col in features:
                ts = convert_to_datetime_seconds(features[self._time_col])
                day = np.array(ts).astype('datetime64[D]')
                features[self._source + '_weekday'] = torch.from_numpy(day.view('int64') % 7 + 1)
                features[self._source + '_hour'] = torch.from_numpy((ts - day).astype('timedelta64[h]').view('int64'))
                month = ts.astype('datetime64[M]')
                features[self._source + '_month'] = torch.from_numpy(month.view('int64') % 12 + 1)
                features[self._source + '_week'] = torch.from_numpy(np.minimum(
                    (ts - month).astype('timedelta64[D]').view('int64') // 7, 4) + 1)
            yield rec