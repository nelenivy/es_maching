import torch
import torch.distributed as dist

from ptls.frames.coles.losses.dist_utils import all_gather_and_cat

class MatchingSoftmaxLoss(torch.nn.Module):
    """
    Softmax loss with the option not taking augmented views of the same sequence into loss
    """
    def __init__(self, temperature=0.05, distributed_mode = False, only_other_mod=False):
        super().__init__()
        
        self.temperature = temperature
        self.distributed_mode = distributed_mode
        self.only_other_mod = only_other_mod
        
    def forward(self, embeddings, classes, mod_segms=None):
        if dist.is_initialized() and self.distributed_mode:
            dist.barrier()
            embeddings = all_gather_and_cat(embeddings)
            classes = classes + (classes.max()+1) * dist.get_rank()
            classes = all_gather_and_cat(classes)
            if self.only_other_mod:
                mod_segms = all_gather_and_cat(mod_segms)
        d = torch.einsum('bh,kh->bk', embeddings, embeddings) / self.temperature
        
        ix_pos = classes.unsqueeze(1) == classes.unsqueeze(0)
        
        if self.only_other_mod:
            ix_pos = torch.bitwise_and(ix_pos, mod_segms.unsqueeze(1) != mod_segms.unsqueeze(0))
            
        ix_pos.fill_diagonal_(0)
        ix_a, ix_pos = ix_pos.nonzero(as_tuple=True)
        
        _, ix_neg = (classes[ix_a].unsqueeze(1) != classes.unsqueeze(0)).nonzero(as_tuple=True)
        ix_all = torch.cat([ix_pos.unsqueeze(1), ix_neg.view(ix_a.size(0), -1)], dim=1)
        
        return -torch.log_softmax(d[ix_a.unsqueeze(1).expand_as(ix_all), ix_all], dim=1)[:, 0].mean()