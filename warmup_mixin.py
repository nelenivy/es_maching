import pytorch_lightning as pl

class WarmupMixin:#(pl.LightningModule):
    def __init__(self,  
                 *args,
                 warmup_steps = 500,
                 initial_lr = 0.001,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        
    def optimizer_step(self, 
                       epoch, 
                       batch_idx, 
                       optimizer, 
                       #optimizer_idx, 
                       optimizer_closure, 
                       ):
        
        optimizer.step(closure = optimizer_closure)
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.initial_lr
        