import os
import torch
import torch.nn as nn
from model.sam_adapter.sam_adapter import SAM_Adapter

def build_model(config, dist_env):
    # 根据配置构建模型

    model = SAM_Adapter(
        inp_size=config.inp_size,
        encoder_mode=config.encoder_mode,
        loss=config.loss
    )

    model = nn.parallel.DistributedDataParallel(
        model.to(dist_env.local_rank),
        device_ids=[dist_env.local_rank],
        output_device=dist_env.local_rank,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=getattr(config, 'bucket_cap_mb', 25),  # 如果配置中没有 bucket_cap_mb，设置默认值25
    )

    return model
