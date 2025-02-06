import unittest
import torch
import os
import argparse
from src.trainer import Trainer, DistributedEnv
from torch.nn.parallel import DistributedDataParallel as DDP


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = torch.nn.Linear(3 * 224 * 224, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TestTrainer(unittest.TestCase):
    def setUp(self):
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        self.dist_env = DistributedEnv()
        self.dist_env.setup(init_method='tcp://127.0.0.1:12355')

        self.config = argparse.Namespace(
            work_dir='.',
            task_name='test_task',
            lr=0.001,
            weight_decay=0.0,
            epochs_per_session=1,
            batch_size=2,
            num_workers=0,
            total_sessions=1,
            resume_from_checkpoint=None,
            print_freq=1,
            inp_size=224,
            encoder_mode={
                'name': 'dummy_encoder',
                'embed_dim': 256,
                'depth': 12,
                'num_heads': 8,
                'mlp_ratio': 4,
                'qkv_bias': True,
                'patch_size': 16,
                'out_chans': 256,
                'use_rel_pos': True,
                'window_size': 7,
                'global_attn_indexes': [1, 3, 5, 7],
                'prompt_embed_dim': 256,
            },
            loss='cross_entropy',
            bucket_cap_mb=25,
        )

        self.trainer = Trainer(self.config, self.dist_env)

        model = DummyModel().to(self.dist_env.device)
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[self.dist_env.local_rank], output_device=self.dist_env.local_rank)
        else:
            model = DDP(model)
        self.trainer.model = model

        self.trainer.optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        self.trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.trainer.optimizer,
            T_max=self.config.epochs_per_session,
            eta_min=getattr(self.config, 'scheduler_eta_min', 1e-6)
        )

        self.trainer.criterion = torch.nn.CrossEntropyLoss()

        # 创建一个简单的数据集和数据加载器
        dummy_data = torch.randn(10, 3, 224, 224, device=self.dist_env.device)
        dummy_targets = torch.randint(0, 10, (10,), device=self.dist_env.device)
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_targets)
        self.trainer.train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        self.trainer.valid_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    def test_train(self):
        # 运行一个训练 epoch
        self.trainer.train_epoch(0, self.trainer.train_loader)
        # 运行验证
        valid_stats = self.trainer.validate()
        # 检查验证的结果
        self.assertIn('accuracy', valid_stats)
        print(f"Validation Accuracy: {valid_stats['accuracy']}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 