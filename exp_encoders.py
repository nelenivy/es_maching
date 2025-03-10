import torch
import numpy as np
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn.trx_encoder.noisy_embedding import NoisyEmbedding
import ciso8601
import time
import math 
from ptls.data_load.padded_batch import PaddedBatch
from resample import resample, add_space_and_session_tokens
from einops import rearrange, repeat
import random
import x_transformer

class SeqEncoderWithPostproc(SeqEncoderContainer):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,                                  
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )
        
        self.col_time=col_time
        
    def post_process_embs(self, x, x_time):
        return x
    
    def forward(self, x):
        #the modification is that there is a possibility of postprocess
        x_time=x.payload[self.col_time]
        x = self.trx_encoder(x)
        x = self.post_process_embs(x, x_time)
        x = self.seq_encoder(x)
        return x
    
class SeqEncoderWithMask(SeqEncoderWithPostproc):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size,
            is_reduce_sequence=is_reduce_sequence,
            col_time=col_time,
            **seq_encoder_params 
        )
        self.use_mask_of_padded = use_mask_of_padded
        self.pass_time_to_encoder = pass_time_to_encoder
        
    def post_process_embs(self, x, x_time):
        return x, x_time
    
    def forward(self, x):
        #the modification is that there is a usage of mask
        x_time=x.payload[self.col_time]
        x = self.trx_encoder(x)
        x, x_time = self.post_process_embs(x, x_time)
        
        if self.use_mask_of_padded:
            mask  = torch.bitwise_and(x_time > 0, x_time != float('inf'))
        else:
            mask = None
            
        x = self.seq_encoder(x, mask=mask, time=x_time if self.pass_time_to_encoder else None)
        return x

class SeqEncoderResample(SeqEncoderWithPostproc):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,                 
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size,
            is_reduce_sequence=is_reduce_sequence,
            col_time=col_time,            
            **seq_encoder_params
        )
        
        self.zero_token = NoisyEmbedding(1, input_size, padding_idx=None, noise_scale=0.003, sparse=False)
        self.pad_token = NoisyEmbedding(1, input_size, padding_idx=None, noise_scale=0.003, sparse=False)        
        
    def post_process_embs(self, mod_batch_feats, mod_batch_time):
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
                interval, normalize=False, wtype='kde', 
                zero_token=self.zero_token,
                pad_token=None, #self.pad_token,
                max_len=ints_num, sigma=0.5, 
                                                   min_time=None,#int(min_time), 
                                                   cut_after_end=True, cut_time_after_time=True)

        mod_batch_feats = PaddedBatch(payload=mod_batch_feats, 
                                      length=(mod_batch_time > 0).sum(dim=1))
        return mod_batch_feats

    
class SeqEncoderAddToken(SeqEncoderWithPostproc):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,                 
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size,
            is_reduce_sequence=is_reduce_sequence,
            col_time=col_time,            
            **seq_encoder_params
        )
        
        self.space_token = NoisyEmbedding(1, input_size, padding_idx=None, noise_scale=0.003, sparse=False)    
        
    def post_process_embs(self, mod_batch_feats, mod_batch_time):
        interval = 12 * 60 * 60#int(math.ceil((max_time - min_time) / ints_num))
        #print(mod_batch_feats.payload.shape[1])
        before_len = mod_batch_feats.payload.shape[1]
        mod_batch_time, mod_batch_feats = add_space_and_session_tokens(mod_batch_feats.payload, mod_batch_time, 
                interval, 0,space_token=self.space_token, session_token=None, sort=True, max_len=280)       

        mod_batch_feats = PaddedBatch(payload=mod_batch_feats, 
                                      length=(mod_batch_time > 0).sum(dim=1))
        #print(before_len, mod_batch_feats.payload.shape[1], (mod_batch_time > 0).sum(dim=1))
        return mod_batch_feats
    
class SeqEncoderInverted(SeqEncoderWithPostproc):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,                 
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size,
            is_reduce_sequence=is_reduce_sequence,
            col_time=col_time,            
            **seq_encoder_params
        )
        
        #self.pad_token = NoisyEmbedding(1, input_size, padding_idx=None, noise_scale=0.003, sparse=False)   
        self.max_size = 256
        
    def post_process_embs(self, batch, mod_batch_time):
        mod_batch_feats = batch.payload
        if mod_batch_feats.shape[1] < self.max_size:
            pad_right = self.max_size - mod_batch_feats.shape[1]
            
            if self.training:
                pad_left = random.randrange(pad_right)
                pad_right = pad_right - pad_left
                padding_left = torch.zeros(
                (mod_batch_feats.shape[0], pad_left, mod_batch_feats.shape[2]),
                    dtype=mod_batch_feats.dtype, device=mod_batch_feats.device)
                mod_batch_feats = torch.concat((padding_left, mod_batch_feats), dim=1)
                
            padding_right = torch.zeros(
                (mod_batch_feats.shape[0], pad_right, mod_batch_feats.shape[2]),
            dtype=mod_batch_feats.dtype, device=mod_batch_feats.device)
            mod_batch_feats = torch.concat((mod_batch_feats, padding_right), dim=1)
            
        mod_batch_feats = torch.permute(mod_batch_feats, (0, 2, 1))        
        mod_batch_feats = PaddedBatch(payload=mod_batch_feats, 
                                      length=self.max_size)
        return mod_batch_feats
    
class SeqEncoderDual(SeqEncoderWithMask):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size,
            is_reduce_sequence=is_reduce_sequence,
            use_mask_of_padded=use_mask_of_padded,
            pass_time_to_encoder=pass_time_to_encoder,
            col_time=col_time, 
            **seq_encoder_params
        )
        
        #self.pad_token = NoisyEmbedding(1, input_size, padding_idx=None, noise_scale=0.003, sparse=False)   
        self.max_size = 256
        
    def post_process_embs(self, batch, mod_batch_time):
        mod_batch_feats = batch.payload
        if mod_batch_feats.shape[1] < self.max_size:
            pad_right = self.max_size - mod_batch_feats.shape[1]
                
            padding_right = torch.zeros(
                (mod_batch_feats.shape[0], pad_right, mod_batch_feats.shape[2]),
            dtype=mod_batch_feats.dtype, device=mod_batch_feats.device)
            mod_batch_feats = torch.concat((mod_batch_feats, padding_right), dim=1)
            
            padding_time = torch.zeros(
                (mod_batch_time.shape[0], 
                 pad_right),
            dtype=mod_batch_time.dtype, device=mod_batch_time.device)
            mod_batch_time  = torch.concat((mod_batch_time, padding_time), dim=1)
               
        mod_batch_feats = PaddedBatch(payload=mod_batch_feats, 
                                      length=self.max_size)
        return mod_batch_feats, mod_batch_time
    
class SeqEncoderPatch(SeqEncoderWithMask):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,                   
                patch_size=4,
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size,
            is_reduce_sequence=is_reduce_sequence,
            use_mask_of_padded=use_mask_of_padded,
            pass_time_to_encoder=pass_time_to_encoder,
            col_time=col_time, 
            **seq_encoder_params
        )
           
        self.patch_size = patch_size
        
    def post_process_embs(self, batch, mod_batch_time):
        mod_batch_feats = batch.payload
        
        if mod_batch_feats.shape[1] % self.patch_size != 0:
            padding = torch.zeros(
                (mod_batch_feats.shape[0], 
                 self.patch_size - mod_batch_feats.shape[1] % self.patch_size, 
                 mod_batch_feats.shape[2]),
            dtype=mod_batch_feats.dtype, device=mod_batch_feats.device)
            mod_batch_feats = torch.concat((mod_batch_feats, padding), dim=1)
            
            padding_time = torch.zeros(
                (mod_batch_time.shape[0], 
                 self.patch_size - mod_batch_time.shape[1] % self.patch_size),
            dtype=mod_batch_time.dtype, device=mod_batch_time.device)
            mod_batch_time  = torch.concat((mod_batch_time, padding_time), dim=1)
        
        mod_batch_feats= rearrange(mod_batch_feats, 'b (l1 l2) w -> b l1 (l2 w) ', l2=self.patch_size) 
        mod_batch_time= rearrange(mod_batch_time, 'b (l1 l2) -> b l1 l2 ', l2=self.patch_size) 
        mod_batch_time = mod_batch_time.sum(dim=2) // mod_batch_time.shape[2]
        mod_batch_feats = PaddedBatch(payload=mod_batch_feats, 
                                      length=(mod_batch_time > 0).sum(dim=1))
        return mod_batch_feats, mod_batch_time
    
class SeqEncoderStacked(SeqEncoderWithPostproc):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,                 
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size,
            is_reduce_sequence=is_reduce_sequence,
            col_time=col_time,            
            **seq_encoder_params
        )
           
        self.bottom_enc = x_transformer.XTransformerEncoder(
            attn_layers=x_transformer.Encoder(dim=input_size,
            depth=2,
            dynamic_pos_bias=True,
            dynamic_pos_bias_log_distance=False),  
            is_reduce_sequence=False,
            input_size=input_size)
        
    def post_process_embs(self, batch, mod_batch_time):
        mask = mod_batch_time.bool()
        mod_batch_feats = self.bottom_enc(batch, mask=mask)
 
        mod_batch_feats = PaddedBatch(payload=mod_batch_feats, 
                                      length=batch.seq_lens)
        return mod_batch_feats
    

class SeqEncoderPatchTransf(SeqEncoderWithMask):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,                   
                patch_size=10,
                 patch_enc_depth=4,
                 mem_len=1,
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size * mem_len,
            is_reduce_sequence=is_reduce_sequence,
            use_mask_of_padded=use_mask_of_padded,
            pass_time_to_encoder=pass_time_to_encoder,
            col_time=col_time, 
            **seq_encoder_params
        )
           
        self.patch_size = patch_size
        self.patch_enc = x_transformer.XTransformerEncoder(
            attn_layers=x_transformer.Encoder(dim=input_size,
            depth=patch_enc_depth,
            dynamic_pos_bias=True,
            dynamic_pos_bias_log_distance=False),  
            is_reduce_sequence=True,
            input_size=input_size,
            reduce_type="all_mem",
            num_memory_tokens=mem_len
        )
        self.num_memory_tokens = mem_len
        
    def post_process_embs(self, batch, mod_batch_time):
        mod_batch_feats = batch.payload
        
        if mod_batch_feats.shape[1] % self.patch_size != 0:
            padding = torch.zeros(
                (mod_batch_feats.shape[0], 
                 self.patch_size - mod_batch_feats.shape[1] % self.patch_size, 
                 mod_batch_feats.shape[2]),
            dtype=mod_batch_feats.dtype, device=mod_batch_feats.device)
            mod_batch_feats = torch.concat((mod_batch_feats, padding), dim=1)
            
            padding_time = torch.zeros(
                (mod_batch_time.shape[0], 
                 self.patch_size - mod_batch_time.shape[1] % self.patch_size),
            dtype=mod_batch_time.dtype, device=mod_batch_time.device)
            mod_batch_time  = torch.concat((mod_batch_time, padding_time), dim=1)
        
        patch_feats= rearrange(mod_batch_feats, 'b (l1 l2) w -> (b l1) l2 w', l2=self.patch_size)
        patch_time= rearrange(mod_batch_time, 'b (l1 l2) -> (b l1) l2 ', l2=self.patch_size) 
        patch_feats = PaddedBatch(payload=patch_feats, 
                                      length=(patch_time > 0).sum(dim=1))
        aggr_patch_feats = self.patch_enc(patch_feats, mask=patch_time.bool())
        aggr_feats = rearrange(aggr_patch_feats, '(b l1) m w -> b l1 (m w)', 
                               b=mod_batch_feats.shape[0], m=self.num_memory_tokens)
        aggr_time= rearrange(mod_batch_time, 'b (l1 l2) -> b l1 l2 ', l2=self.patch_size) 
        aggr_time = aggr_time.sum(dim=2) // aggr_time.shape[2]
        aggr_feats = PaddedBatch(payload=aggr_feats, 
                                      length=(aggr_time > 0).sum(dim=1))
        return aggr_feats, aggr_time
    
class SeqEncoderElemPatch(SeqEncoderWithMask):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,                   
                 patch_size=4,
                 stride=1,
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size * patch_size,
            is_reduce_sequence=is_reduce_sequence,
            use_mask_of_padded=use_mask_of_padded,
            pass_time_to_encoder=pass_time_to_encoder,
            col_time=col_time, 
            **seq_encoder_params
        )
           
        self.patch_size = patch_size
        self.stride = stride
        
    def post_process_embs(self, batch, mod_batch_time):
        mod_batch_feats = batch.payload
        init_seq_len = mod_batch_feats.shape[1]
        #add padding
        padding_left = torch.zeros(
            (mod_batch_feats.shape[0], 
             math.floor(self.patch_size / 2), 
             mod_batch_feats.shape[2]),
        dtype=mod_batch_feats.dtype, device=mod_batch_feats.device)
        
        padding_right = torch.zeros(
            (mod_batch_feats.shape[0], 
             math.ceil(self.patch_size / 2), 
             mod_batch_feats.shape[2]),
        dtype=mod_batch_feats.dtype, device=mod_batch_feats.device)
        
        mod_batch_feats = torch.concat((padding_left, mod_batch_feats, padding_right), dim=1)
        
        shifts = torch.arange(self.patch_size)
        coords = torch.arange(0, init_seq_len, self.stride)
        patch_coords = rearrange(coords, "n -> n 1") + rearrange(shifts, "n -> 1 n")
        #print(patch_coords)
        #patch_coords = repeat(patch_coords, "l p -> b l p w", b=mod_batch_feats.shape[0], w=mod_batch_feats.shape[2])
        #print(patch_coords)
        patched_feats = mod_batch_feats[:, patch_coords, :].clone()
        #print(patched_feats.shape)
        patched_feats = rearrange(patched_feats, 'b l p w -> b l (p w)') 
        #print(patched_feats.shape)   
        patch_times = mod_batch_time[:, : :self.stride]
        patch_seq_lens = (patch_times > 0).sum(dim=1)
        mod_batch_feats = PaddedBatch(payload=patched_feats, 
                                      length=patch_seq_lens)
        return mod_batch_feats, patch_times
    
class SeqEncoderElemPatchTransf(SeqEncoderWithMask):
    def __init__(self,
                 seq_encoder_cls,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=True,
                 use_mask_of_padded=False,
                 pass_time_to_encoder=False,                   
                 patch_size=4,
                 stride=1,
                 patch_enc_depth=4,
                 mem_len=1,
                 col_time='event_time',
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=seq_encoder_cls,
            input_size=input_size,# * mem_len,
            is_reduce_sequence=is_reduce_sequence,
            use_mask_of_padded=use_mask_of_padded,
            pass_time_to_encoder=pass_time_to_encoder,
            col_time=col_time, 
            **seq_encoder_params
        )
           
        self.patch_size = patch_size
        self.stride = stride
        
        self.patch_enc = x_transformer.XTransformerEncoder(
            attn_layers=x_transformer.Encoder(dim=input_size,
            depth=patch_enc_depth,
            dynamic_pos_bias=True,
            dynamic_pos_bias_log_distance=False),  
            is_reduce_sequence=True,
            input_size=input_size,
            reduce_type="all_mem",
            num_memory_tokens=mem_len
        )
        self.num_memory_tokens = mem_len
        
    def post_process_embs(self, batch, mod_batch_time):
        mod_batch_feats = batch.payload
        init_seq_len = mod_batch_feats.shape[1]
        #add padding
        padding_left = torch.zeros(
            (mod_batch_feats.shape[0], 
             math.floor(self.patch_size / 2), 
             mod_batch_feats.shape[2]),
        dtype=mod_batch_feats.dtype, device=mod_batch_feats.device)
        
        padding_right = torch.zeros(
            (mod_batch_feats.shape[0], 
             math.ceil(self.patch_size / 2), 
             mod_batch_feats.shape[2]),
        dtype=mod_batch_feats.dtype, device=mod_batch_feats.device)
        
        mod_batch_feats = torch.concat((padding_left, mod_batch_feats, padding_right), dim=1)
        
        padding_left_time = repeat(mod_batch_time[:, 0], "b -> b l", l=math.floor(self.patch_size / 2))
        padding_right_time = repeat(mod_batch_time[:, 0], "b -> b l", l=math.ceil(self.patch_size / 2))        
        mod_batch_time = torch.concat((padding_left_time, mod_batch_time, padding_right_time), dim=1)
        
        shifts = torch.arange(self.patch_size)
        coords = torch.arange(0, init_seq_len, self.stride)
        patch_coords = rearrange(coords, "n -> n 1") + rearrange(shifts, "n -> 1 n")
        #print(patch_coords)
        patched_feats = mod_batch_feats[:, patch_coords, :].clone()
        #print(patched_feats.shape)
        #patched_feats = rearrange(patched_feats, 'b l p w -> b l (p w)') 
        #print(patched_feats.shape)   
        patch_times = mod_batch_time[:, patch_coords].clone()
        feats_in_elems = rearrange(patched_feats, 'b l p w -> (b l) p w') 
        times_in_elems = rearrange(patch_times, 'b l p -> (b l) p') 
        feats_in_elems = PaddedBatch(payload=feats_in_elems, 
                                      length=(times_in_elems > 0).sum(dim=1))
        aggr_elem_feats = self.patch_enc(feats_in_elems, mask=times_in_elems.bool())
        aggr_elem_feats = rearrange(aggr_elem_feats, '(b l) m w -> b (l m) w', 
                               b=mod_batch_feats.shape[0], m=self.num_memory_tokens)
        aggr_elem_time= patch_times.sum(dim=2) // patch_times.shape[2]
        aggr_elem_time = repeat(aggr_elem_time, "b l -> b (l m)", m=self.num_memory_tokens)
        #print(aggr_elem_time)
        aggr_elem_feats = PaddedBatch(payload=aggr_elem_feats, 
                                      length=(aggr_elem_time > 0).sum(dim=1))        
        
        return aggr_elem_feats, aggr_elem_time
    
class TLBEEncoder(torch.nn.Module):
    def __init__(self, input_size, is_reduce_sequence=False, depth=4, r=2, memory_len=4, chunk_size=10, dim=128):
        super(TLBEEncoder, self).__init__()
        self.perc_depth = depth
        self.r=r
        self.memory_len = memory_len
        self.chunk_size = chunk_size
        block_list = ''
        
        for i in range(self.perc_depth):
            if (i  + 1) % self.r != 0:
                block_list += 'af'
            else:
                block_list += 'cf'
        
        attn_layers =x_transformer.Encoder(dim=dim,
                                           depth=self.perc_depth,
            dynamic_pos_bias=True,
            dynamic_pos_bias_log_distance=False,
            cross_attend = True,
            only_cross = False,
            custom_layers = block_list)
                                           
        self.fast_stream = x_transformer.XTransformerEncoder(input_size=input_size,        
            attn_layers=attn_layers,
            num_memory_tokens=1,
            return_last = False,
            post_emb_norm = False,
            is_reduce_sequence = False)
        
        self.slow_stream = x_transformer.Encoder(dim=dim,
                                                 depth=1,
            dynamic_pos_bias=True,
            dynamic_pos_bias_log_distance=False,
            cross_attend = True,
            only_cross = True,
            custom_layers = "cf")
        
        self.memory = NoisyEmbedding(self.memory_len, dim, padding_idx=None, noise_scale=0.003, sparse=False)       
        
        
    def forward_one_chunk(self, x, context, mask=None):  
        x = self.fast_stream(PaddedBatch(x, length=None), mask=mask, context=context)
        context = self.slow_stream(context, context=x, context_mask=mask)
        
        return context
    
    def forward(self, x, mask=None, time=None):
        N, S0, _ = x.payload.size()

        chunk_num = math.ceil(S0 / self.chunk_size)

        chunked_sequence = torch.chunk(x.payload, chunks=chunk_num, dim=1)
        assert (len(chunked_sequence) == chunk_num)
        chunked_input_mask = torch.chunk(mask, chunks=chunk_num, dim=1)
        mem_ids = torch.arange(self.memory_len, device=x.device)
        mem_ids = repeat(mem_ids, "l -> b l", b=N)
        memory = self.memory(mem_ids)        

        for sequence, input_mask in zip(chunked_sequence, chunked_input_mask):
            if input_mask.sum() > 0:
                memory = self.forward_one_chunk(sequence, memory, mask=input_mask)
                
        return memory.mean(dim=1)