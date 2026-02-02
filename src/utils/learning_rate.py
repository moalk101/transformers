import math
from torch.optim.lr_scheduler import LRScheduler

class TransformerLR(LRScheduler):
    
    def __init__(self, optimizer,d_model,warmup_steps, last_epoch = -1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        step = max(1,self.last_epoch + 1)
        lr = self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        lrs = [lr for _ in self.optimizer.param_groups] 
        
        return lrs