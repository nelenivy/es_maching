import torch
import torch.distributed as dist

class MipNCELoss(torch.nn.Module):
    """ from UNiCLIP https://arxiv.org/pdf/2209.13430
    All positive pairs as `anchor`-`pos_sample` concated with all possible `neg_samples` from batch.
    Softmax is applied ower `distances` between `anchor` ans pos or neg samples.
    `distances` is a dot product.
    Loss minimize `-log()` for `anchor-pos` position.
    
    Params:
        temperature:
            `softmax(distances / temperature)` - scale a sub-exponent expression.
            default 0.05 value is for l2-normalized `embeddings` where dot product distance is in range [-1, 1]
    """
    def __init__(self, temperature=0.05, distributed_mode = False):
        super().__init__()
        
        self.temperature = temperature
        self.distributed_mode = distributed_mode
        
    def forward(self, embeddings, classes):
        d = torch.einsum('bh,kh->bk', embeddings, embeddings) / self.temperature
        
        pos_bin = classes.unsqueeze(1) == classes.unsqueeze(0)
        ix_a, ix_pos = pos_bin.nonzero(as_tuple=True)        
        pos_dists = d[ix_a, ix_pos].clone().requires_grad_(True).view(ix_a.size(0), -1)        

        pos_bin = (classes[ix_a].unsqueeze(1) == classes.unsqueeze(0))
        d_for_calc = d[ix_a, :]
        d_for_calc[pos_bin] = torch.finfo(d.dtype).min
        all_dists = torch.cat([pos_dists, d_for_calc], dim=1)
        
        return -torch.log_softmax(all_dists, dim=1)[:, 0].sum()