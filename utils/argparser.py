

import argparse
from dataclasses import dataclass

@dataclass
class BaseConfig:
    task_name: str = "SAFIRE"
    model_type: str = "vit_b_adaptor"
    checkpoint: str = "sam_vit_b_01ec64.pth"
    work_dir: str = "./work_dir"
    num_epochs: int = 150
    num_pairs: int = 3
    lambda_ps: float = 0.1
    batch_size: int = 8
    num_workers: int = 8
    weight_decay: float = 0.01
    lr: float = 0.0001
    bucket_cap_mb: int = 25
    resume: str = ""
    encresume: str = ""
    init_method: str = "env://"
    wandb_project: str = "SAFIRE"
    wandb_entity: str = None  # Your wandb username/organization
    wandb_name: str = None    # Run name, will default to timestamp if None
    wandb_mode: str = "online"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="SAFIRE")
    parser.add_argument("--model_type", type=str, default="vit_b_adaptor")
    parser.add_argument("--checkpoint", type=str, default="sam_vit_b_01ec64.pth")
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--num_pairs", type=int, default=3)
    parser.add_argument("--lambda_ps", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--bucket_cap_mb", type=int, default=25)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--encresume", type=str, default="")
    parser.add_argument("--init_method", type=str, default="env://")
    parser.add_argument("--wandb_project", type=str, default="SAFIRE")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    
    args = parser.parse_args()
    return BaseConfig(**vars(args))
