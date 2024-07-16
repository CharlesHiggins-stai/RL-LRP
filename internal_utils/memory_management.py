import torch 
import wandb
import gc

def log_memory_usage(wandb_log = False):
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    if wandb_log:
        wandb.log({"device/)memory_summary": torch.cuda.memory_summary(device=None, abbreviated=False)})

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
