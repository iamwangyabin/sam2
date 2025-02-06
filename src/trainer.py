# -*- coding: utf-8 -*-

import os
import gc
from datetime import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from torch.utils.data.distributed import DistributedSampler
from data.datamanger import IncrementalDataManager

from src.build_model import build_model
from utils.logger import Logger


class DistributedEnv:
    def __init__(self):
        self.global_rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.is_main_host = self.global_rank == 0

    def setup(self, init_method):
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=self.global_rank,
            world_size=self.world_size
        )

class Trainer:
    def __init__(self, config, dist_env):
        self.config = config
        self.dist_env = dist_env

        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_save_path = os.path.join(config.work_dir, f"{config.task_name}-{self.run_id}")
        if self.dist_env.is_main_host:
            os.makedirs(self.model_save_path, exist_ok=True)

        self.logger = Logger(config, dist_env, self.run_id)

        self.model = build_model(config, dist_env)
        self.optimizer, self.scheduler = self.setup_optimization()

        self.best_valid_metrics = initialize_metrics()
        self.metric_history = {metric: [] for metric in self.best_valid_metrics}
        self.top_k = getattr(config, 'save_top_k', 3)

        self.total_sessions = getattr(config, "total_sessions", 1)
        self.current_session = 0

        # 初始化起始 epoch
        self.start_epoch = 0

        # 如果指定了检查点路径，加载检查点
        if config.resume_from_checkpoint is not None:
            self.load_checkpoint(config.resume_from_checkpoint)

        self.criterion = get_losses(config)  # 确保定义损失函数

    def setup_optimization(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in params)
        if self.dist_env.is_main_host:
            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs_per_session,
            eta_min=getattr(self.config, 'scheduler_eta_min', 1e-6)
        )

        try:
            from thop import profile
            input_sample = torch.randn(1, 3, 224, 224).to(next(self.model.parameters()).device)
            macs, _ = profile(self.model, inputs=(input_sample, ))
            if self.dist_env.is_main_host:
                print(f"MACs: {macs}")  # 直接打印 MACs
        except ImportError:
            if self.dist_env.is_main_host:
                print("Please make sure to install thop library.")

        return optimizer, scheduler

    def validate(self):
        self.model.eval()
        valid_stats = {
            'loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0,
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_loader):
                images = batch[0].cuda(non_blocking=True)
                targets = batch[1].cuda(non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # 计算预测结果
                _, preds = torch.max(outputs, 1)
                correct = (preds == targets).sum().item()
                total = targets.size(0)

                valid_stats['loss'] += loss.item()
                valid_stats['correct'] += correct
                valid_stats['total'] += total
                valid_stats['batch_count'] += 1

        # 汇总所有进程的验证指标
        for key in ['loss', 'correct', 'total', 'batch_count']:
            tensor = torch.tensor(valid_stats[key]).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            valid_stats[key] = tensor.item()

        # 计算平均损失和准确率
        valid_stats['loss'] /= valid_stats['batch_count']
        valid_stats['accuracy'] = valid_stats['correct'] / valid_stats['total']

        # 记录验证指标
        if self.dist_env.is_main_host:
            self.logger.log_metrics(
                {
                    'valid/loss': valid_stats['loss'],
                    'valid/accuracy': valid_stats['accuracy'],
                },
                step=self.current_epoch
            )

            print(f"Validation Loss: {valid_stats['loss']:.4f}")
            print(f"Validation Accuracy: {valid_stats['accuracy']:.4f}")

        return valid_stats

    def _get_session_loaders(self, session_id):
        # 创建数据管理器，获取当前会话的数据集
        data_manager = IncrementalDataManager(self.config)
        data_manager.set_session(session_id)

        # 获取训练和验证数据集
        train_dataset = data_manager.get_train_dataset()
        valid_dataset = data_manager.get_valid_dataset()

        # 设置采样器（如果是分布式环境）
        if self.dist_env.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.dist_env.world_size,
                rank=self.dist_env.global_rank
            )
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=self.dist_env.world_size,
                rank=self.dist_env.global_rank
            )
        else:
            train_sampler = None
            valid_sampler = None

        # 创建训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        # 创建验证数据加载器
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        return train_loader, valid_loader

    def train_epoch(self, epoch, train_loader):
        """训练一个 epoch"""
        self.model.train()
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        epoch_stats = {
            'loss': 0.0,
            'batch_count': 0,
            # 根据需要添加其他统计指标
        }

        for batch_idx, batch in enumerate(train_loader):
            images = batch[0].cuda(non_blocking=True)
            targets = batch[1].cuda(non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            loss = self.criterion(outputs, targets) 
            
            loss.backward()

            if hasattr(self.config, 'grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.grad_clip
                )
            
            self.optimizer.step()
            
            epoch_stats['loss'] += loss.item()
            epoch_stats['batch_count'] += 1
            
            # 定期打印训练信息
            if batch_idx % self.config.print_freq == 0 and self.dist_env.is_main_host:
                print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}')

        # 汇总所有进程的训练损失
        tensor = torch.tensor([epoch_stats['loss'], epoch_stats['batch_count']]).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        total_loss = tensor[0].item()
        total_batches = tensor[1].item()
        epoch_stats['loss'] = total_loss / total_batches
        
        # 记录训练指标
        if self.dist_env.is_main_host:
            self.logger.log_metrics(
                {'train/loss': epoch_stats['loss']},
                step=epoch
            )

        return epoch_stats

    def train(self):
        """训练主循环"""
        for session_id in range(self.current_session, self.total_sessions):
            self.current_session = session_id
            train_loader, valid_loader = self._get_session_loaders(session_id)
            
            # 设置当前会话的数据加载器
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            
            if self.dist_env.is_main_host:
                print(f"\nStarting session {session_id}")
                print(f"Training samples: {len(train_loader.dataset)}")
                print(f"Validation samples: {len(valid_loader.dataset)}")

            for epoch in range(self.start_epoch, self.config.epochs_per_session):
                self.current_epoch = epoch  # 添加这一行
                # 训练一个 epoch
                train_stats = self.train_epoch(epoch, train_loader)
                
                # 验证
                valid_stats = self.validate()
                
                # 更新学习率
                self.scheduler.step()
                
                # 记录当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_metrics({'train/lr': current_lr}, step=epoch)
                
                # 保存检查点
                self.save_checkpoint(epoch, valid_stats)
                
                # 打印进度信息
                if self.dist_env.is_main_host:
                    print(f"\nEpoch {epoch} Summary:")
                    print(f"Training Loss: {train_stats['loss']:.4f}")
                    print(f"Validation Loss: {valid_stats['loss']:.4f}")
                    print(f"Learning Rate: {current_lr:.6f}")
                    
                # 修改清理内存的部分
                gc_freq = getattr(self.config, 'gc_freq', None)
                if gc_freq is not None and epoch % gc_freq == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # 重置起始 epoch，为下一个会话做准备
            self.start_epoch = 0
            
            if self.dist_env.is_main_host:
                print(f"\nCompleted session {session_id}")

    def load_checkpoint(self, checkpoint_path):
        # 在分布式环境中映射到正确的设备
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.dist_env.local_rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_session = checkpoint.get('session', self.current_session)
        self.start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个 epoch 开始
        if self.dist_env.is_main_host:
            print(f"Resumed training from checkpoint: {checkpoint_path}")
            print(f"Resuming from session {self.current_session}, epoch {self.start_epoch}")

    def save_checkpoint(self, epoch, valid_stats):
        if self.dist_env.is_main_host:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'session': self.current_session,
                'epoch': epoch,
                'valid_stats': valid_stats,
                # 如果有其他需要保存的信息，请一并添加
            }
            checkpoint_path = os.path.join(self.model_save_path, f'checkpoint_session{self.current_session}_epoch{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")








