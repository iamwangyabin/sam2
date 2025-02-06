import os
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from data.datamanger import IncrementalDataManager

manager = IncrementalDataManager("configs/train.yaml")
for session_id in manager.session_datasets.keys():
    manager.set_session(session_id)
    dataloader = DataLoader(manager, batch_size=12, shuffle=True, num_workers=0)
    for batch_idx, data in enumerate(dataloader):
        print(f"Session: {session_id}, Batch: {batch_idx}")

        # data 可能包含 3 or 4 个元素, 视具体返回结构而定
        if len(data) == 3:
            # 假设 data 为 (images, masks, paths)
            img_tensor, mask_tensor, paths = data
            # 可视化 batch 中第一个样本
            img_np = img_tensor[0].permute(1, 2, 0).numpy().astype(np.uint8)

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(img_np)
            axs[0].set_title(f"Image [0]")
            axs[0].axis("off")

            # mask_tensor[0] : shape (H, W) 或 (1, H, W)
            if mask_tensor[0].dim() == 3:
                axs[1].imshow(mask_tensor[0][0].numpy(), cmap='gray')
            else:
                axs[1].imshow(mask_tensor[0].numpy(), cmap='gray')
            axs[1].set_title("Mask [0]")
            axs[1].axis("off")

            plt.suptitle(f"Path: {paths[0]}")
            plt.tight_layout()
            plt.show()

        elif len(data) == 4:
            # 假设 data 为 (images, masks, points, paths)
            img_tensor, mask_tensor, point_tensor, paths = data
            # 可视化 batch 中第一个样本
            img_np = img_tensor[0].permute(1, 2, 0).numpy().astype(np.uint8)
            n_channels = mask_tensor.shape[1]  # shape 通常是 (B, C, H, W)

            fig, axs = plt.subplots(1, n_channels + 1, figsize=(6 * (n_channels + 1), 6))
            axs[0].imshow(img_np)
            axs[0].set_title(f"Image [0]")
            axs[0].axis("off")

            # 可视化第一个样本的点信息
            points = point_tensor[0].numpy()  # shape (C, 2) 或 (1, 2)
            if len(points.shape) == 2:
                for i in range(points.shape[0]):
                    x, y = points[i]
                    axs[0].scatter(x, y, c='red', s=40)

            # 依次可视化各通道的 mask
            for c in range(n_channels):
                axs[c + 1].imshow(mask_tensor[0, c].numpy(), cmap='gray')
                axs[c + 1].set_title(f"Mask channel={c} [0]")
                axs[c + 1].axis("off")

            plt.suptitle(f"Path: {paths[0]}")
            plt.tight_layout()
            plt.show()

        # 演示只可视化第一个 batch
        break  
