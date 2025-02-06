import unittest
import torch
import os
import argparse
from src.trainer import Trainer, DistributedEnv

# 创建一个空的模型用于测试
class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

    def forward(self, x):
        # 返回一个固定的张量作为输出
        return torch.zeros(x.size(0), 10).cuda()  # 假设有10个类别

class TestTrainer(unittest.TestCase):
    def setUp(self):
        # 设置必要的环境变量，模拟分布式环境
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

        # 初始化分布式环境
        self.dist_env = DistributedEnv()
        self.dist_env.setup(init_method='env://')

        # 创建配置对象
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
        )

        # 初始化Trainer并替换模型为空模型
        self.trainer = Trainer(self.config, self.dist_env)
        self.trainer.model = DummyModel().cuda()
        self.trainer.optimizer, self.trainer.scheduler = self.trainer.setup_optimization()
        self.trainer.criterion = torch.nn.CrossEntropyLoss()
        
        # 创建一个简单的数据集和数据加载器
        dummy_data = torch.randn(10, 3, 224, 224)
        dummy_targets = torch.randint(0, 10, (10,))
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
    unittest.main() 