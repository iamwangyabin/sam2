
import os
import gc
from datetime import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import monai
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import wandb

from segment_anything import sam_model_registry
import forgery_data_core
from networks.safire_model import AdaptorSAM
import ForensicsEval as FE
from losses import ILW_BCEWithLogitsLoss, PixelAccWithIgnoreLabel







def main():
    torch.manual_seed(2024)
    torch.cuda.empty_cache()
    
    config = parse_args()
    dist_env = DistributedEnv()
    dist_env.setup(config.init_method)
    
    trainer = SAFIRETrainer(config, dist_env)
    
    try:
        train_losses = []
        for epoch in range(config.num_epochs):
            gc.collect()
            
            # Train
            train_stats = trainer.train_epoch(epoch)
            train_losses.append(train_stats['loss'])
            
            # Step scheduler
            trainer.scheduler.step()
            
            # Sync processes
            torch.distributed.barrier()
            
            # Validate
            valid_stats = trainer.validate()
            
            # Save checkpoint
            trainer.save_checkpoint(epoch, valid_stats)
            
            # Sync processes
            torch.distributed.barrier()

    finally:
        if dist_env.is_main_host:
            wandb.finish()

if __name__ == "__main__":
    main()
