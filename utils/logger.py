import os
import wandb
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, config, dist_env, run_id):
        self.dist_env = dist_env
        self.config = config
        self.logger = None

        if self.dist_env.is_main_host:
            if config.logger_type == 'wandb':
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=config.wandb_name or run_id,
                    mode=config.wandb_mode,
                    config=vars(config)
                )
                self.logger = wandb
            elif config.logger_type == 'tensorboard':
                self.logger = SummaryWriter(log_dir=config.tensorboard_log_dir)
            elif config.logger_type == 'custom':
                # 初始化你的自定义日志器
                pass
            else:
                # 不使用任何日志工具
                self.logger = None

    def log_metrics(self, metrics, step=None, commit=True):
        if self.dist_env.is_main_host and self.logger is not None:
            if self.config.logger_type == 'wandb':
                self.logger.log(metrics, step=step, commit=commit)
            elif self.config.logger_type == 'tensorboard':
                for key, value in metrics.items():
                    self.logger.add_scalar(key, value, step)
            elif self.config.logger_type == 'custom':
                # 实现自定义的日志记录
                pass


