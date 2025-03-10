import numpy as np
import torch
import math

def get_tensor_by_indices(tensor, indices):
        batch_size = tensor.shape[0]
        return tensor[:, indices, :][torch.arange(batch_size), torch.arange(batch_size), :, :]
    
def create_weights(resampled_times, times, interval, wtype='kde', sigma=30.0):
    if wtype == 'sum':
        weights = torch.bitwise_and(
            times >= resampled_times - interval // 2,
            times <= resampled_times + interval // 2)
    elif wtype == 'fdistr':
        weights = torch.bitwise_and(times > 0, times <= resampled_times + interval // 2)
    elif wtype == 'kde':        
        time_diff = ((times - resampled_times).to(torch.float32) / interval)
        weights = 1.0 / (math.sqrt(2.0 * math.pi) * sigma) * torch.exp(-time_diff ** 2 / (2.0 * sigma** 2))
    else:
        raise NotImplementedError(wtype)
        
    return weights.unsqueeze(3)

from datetime import datetime

def resample(features, times, interval, normalize=False, zero_token=None, pad_token=None, 
             wtype='kde', sigma=30.0, cut_after_end=True, cut_time_after_time=False,
            max_len=None, min_time=None):
    #case if time is in microseconds
    times[times>9999999999] //= 1000
    times[times>9999999999] //= 1000
    if min_time and min_time > 9999999999:
        min_time //= 1000
    
    #time borders
    times_max = torch.max(times, dim=1, keepdim=True).values
    zero_times_mask = (times == 0)
    times[zero_times_mask] = torch.max(times_max)
    times_min = torch.min(times, dim=1, keepdim=True).values
    left_border_time = times_min
    #print(datetime.utcfromtimestamp(times_min.cpu().numpy().min()).strftime('%Y-%m-%d %H:%M:%S'))
    #print(datetime.utcfromtimestamp(times_max.cpu().numpy().max()).strftime('%Y-%m-%d %H:%M:%S'))
    times[zero_times_mask] = 0
    
    if min_time is None:
        max_step = math.ceil(torch.max(times_max - times_min) / interval)
        #lens = torch.minimum(torch.ceil((times_max - times_min) / interval), torch.tensor(max_len).to('cuda'))
        #lens_inv = (1.0 / lens).squeeze(1)
    
        if max_len:
            max_step = min(max_step, max_len)
            
    elif not(min_time is None) and not (max_len is None):
        times_min = torch.ones_like(times[:, 0].unsqueeze(1)) * min_time
        max_step = max_len
        
    #print((times[times > 0] > min_time).to(float).mean())
    times_diff = torch.arange(max_step, dtype=times.dtype, device=times.device) * interval + interval // 2
    resampled_times = times_diff.unsqueeze(0) + times_min
    #print(resampled_times)
    #print(times)
    resampled_times_for_weights = torch.permute(resampled_times, (1, 0)).unsqueeze(2)
    weights = create_weights(resampled_times_for_weights, times.unsqueeze(0), interval, wtype=wtype, sigma=sigma)
    resampled_feats = (features.unsqueeze(0) * weights).sum(dim=2, keepdim=False)
    resampled_feats = torch.permute(resampled_feats, (1, 0, 2))

    if normalize or not(zero_token is None):            
        non_zero_elems_num = weights.sum(dim=2, keepdim=False)
        non_zero_elems_num = torch.permute(non_zero_elems_num, (1, 0, 2))
    
    if (not (pad_token is None)) or (not (zero_token is None)):
        emb_ids = torch.zeros_like(resampled_times)
        
    if cut_after_end:
        max_time_mask = torch.bitwise_or(
            resampled_times > times_max + interval // 2, 
            resampled_times < left_border_time - interval // 2)  
        #resampled_times > times_max + interval // 2
            #torch.bitwise_or(
            #resampled_times > times_max + interval // 2, 
            #resampled_times < left_border_time - interval // 2)   
        if cut_time_after_time:
            resampled_times[max_time_mask] = 0
          
        before_max_time = torch.bitwise_not(max_time_mask)
        resampled_feats *= before_max_time.unsqueeze(2)
        
        if not (pad_token is None):
            adding_token = pad_token(emb_ids)
            resampled_feats += adding_token * max_time_mask.unsqueeze(2)

    if not(zero_token is None): 
        zero_elems_mask = (non_zero_elems_num == 0)
        
        if cut_after_end:
            zero_elems_mask = torch.bitwise_and(zero_elems_mask, before_max_time.unsqueeze(2)) 

        adding_token = zero_token(emb_ids)
        resampled_feats += adding_token * zero_elems_mask   
        #nums_fill += (zero_elems_mask * lens_inv).sum()
        #nums_no_fill += (before_max_time.squeeze(1).to(float) * lens_inv).sum()
        #nums_pad += (curr_feats.shape[0] - before_max_time_feats.shape[0]) / float(curr_feats.shape[0])
        
    if normalize:
        non_zero_elems_num += (non_zero_elems_num == 0)
        resampled_feats /= non_zero_elems_num
        
    #print(lens.to(float).mean(), torch.median(lens))
    #print(nums_fill / resampled_feats.shape[0], (nums_no_fill - nums_fill)/ resampled_feats.shape[0], nums_pad)
    #print(resampled_times.shape, resampled_feats.shape)
    #print(resampled_feats)
    #print("features")
    #from collections import Counter
    #print(Counter([features[0, i, :].sum().detach().cpu().item() for i in range(0, features.shape[1])]).most_common())
    #print("---------------")
    #print("resampled_feats")
    #print(Counter([resampled_feats[0, i, :].sum().detach().cpu().item() for i in range(0, resampled_feats.shape[1])]).most_common())
    #print("---------------")
    return resampled_times, resampled_feats

def add_space_and_session_tokens(features, times, interval_space, interval_session, 
                                 space_token=None, session_token=None, sort=False, max_len=-1):
    #case if time is in microseconds
    times[times>9999999999] //= 1000
    times[times>9999999999] //= 1000
    
    times_diff = torch.abs(times[:, :-1] - times[:, 1:])
    spaces_mask = times_diff > interval_space
    spaces_time = ((times[:, :-1] + times[:, 1:]) // 2) * spaces_mask
    spaces_num = spaces_mask.sum(dim=1, keepdim=True)
    max_spaces_num = spaces_num.max().item()
    
    if max_len > 0:
        max_diff = max(0, max_len - features.shape[1])
        max_spaces_num = min(max_diff, max_spaces_num)
        
    spaces_time = torch.topk(spaces_time, max_spaces_num, dim=1).values
    embs_ids = torch.zeros_like(spaces_time)
    spaces_to_add = space_token(embs_ids)
    appended_features = torch.cat((features, spaces_to_add), dim=1)
    appended_times = torch.cat((times, spaces_time), dim=1)
    
    if sort:
        max_val = torch.max(appended_times)
        appended_times[appended_times == 0] = max_val + 1
        appended_times, indices_time = torch.sort(appended_times, dim=1)
        appended_features = get_tensor_by_indices(appended_features, indices_time)
        appended_times[appended_times > max_val] = 0        
    
    return appended_times, appended_features