import matplotlib.pyplot as plt
import numpy as np
import torch

from data.database import CoreDataset, FileDataset


def test_all_modes():
    """
    针对 CoreDataset 中支持的所有 mode，依次进行测试并绘制结果。
    注意：
     1) 不同的 mode 返回的数据形状不同，需要分别处理。
     2) 下面的示例代码仅展示如何可视化和检查返回的数据；实际使用时可根据需要做进一步的处理。
    """

    # 1. 创建最基本的 FileDataset
    file_dataset = FileDataset(
        root_path=r"D:\BaiduNetdiskDownload\datasets\tampCOCO",
        im_list_file="dataset_configs/bcm_COCO_list.txt"
    )

    # 2. 定义 CoreDataset 可以使用的所有 mode
    all_modes = [
        "test_auto",
        "im_mask",
        "train",
        "valid",
        "one_random",
        "mid",
        "one_mid",
        "complete_random",
        "pair_random",
        "pair_random_cc"
    ]

    # 3. 选择一个示例索引
    test_index = 201

    for mode in all_modes:
        print(f"\n=== Testing mode: {mode} ===")
        # 对于含有 pair_random 的模式，如果想可视化多个通道，可调整 num_pairs
        # 这里示例 num_pairs=1，这时返回的 mask 可能是 2 通道（正负例）
        dataset = CoreDataset(
            [file_dataset],
            mode=mode,
            num_pairs=1,    # 只采样一个正负对
        )

        # 4. 从 dataset 获取对应的 sample
        #    不同的 mode，返回的数据结构会略有不同
        item = dataset[test_index]

        # 5. 针对返回进行可视化 (分情况处理)
        #    test_auto => (img, mask_tensor, path)
        #    im_mask   => (img_tensor, mask_tensor, path)
        #    其余模式  => (img_tensor, mask_tensor, point_tensor, path)
        if mode == "test_auto":
            # item: (img, mask_tensor, path)
            img, mask_tensor, img_path = item
            # img: shape (H, W, 3), float32
            # mask_tensor: shape (1, H, W), int64
            # 可视化
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(img.astype(np.uint8))  # 转为 uint8 便于可视化
            axs[0].set_title(f"Image\nMode: {mode}")
            axs[0].axis("off")

            axs[1].imshow(mask_tensor[0].numpy(), cmap='gray')
            axs[1].set_title("Mask (channel=0)")
            axs[1].axis("off")

            plt.suptitle(f"Path: {img_path}")
            plt.tight_layout()
            plt.show()

        elif mode == "im_mask":
            # item: (img_tensor, mask_tensor, path)
            img_tensor, mask_tensor, img_path = item
            # img_tensor: shape (3, H, W), float32
            # mask_tensor: shape (H, W) or (1, H, W), int64
            # 可视化
            img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(img_np)
            axs[0].set_title(f"Image\nMode: {mode}")
            axs[0].axis("off")

            if len(mask_tensor.shape) == 3:
                # 如果是 (1, H, W)，只取第 0 维可视化
                axs[1].imshow(mask_tensor[0].numpy(), cmap='gray')
            else:
                # 如果是 (H, W)，直接可视化
                axs[1].imshow(mask_tensor.numpy(), cmap='gray')
            axs[1].set_title("Mask")
            axs[1].axis("off")

            plt.suptitle(f"Path: {img_path}")
            plt.tight_layout()
            plt.show()

        else:
            # item: (img_tensor, mask_tensor, point_tensor, path)
            img_tensor, mask_tensor, point_tensor, img_path = item
            # img_tensor: shape (3, H, W), float32
            # mask_tensor: shape (N, H, W), int64 (N 可能是 1，也可能是 2 或更多通道)
            # point_tensor: shape (N, 2) 或 (1, 2)，float32，表示采样点
            # 可视化
            img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            n_channels = mask_tensor.shape[0]

            fig, axs = plt.subplots(1, n_channels + 1, figsize=(6 * (n_channels + 1), 6))

            # 第一个子图：显示原图并画上采样点
            axs[0].imshow(img_np)
            axs[0].set_title(f"Image with Points\nMode: {mode}")
            axs[0].axis("off")

            # 如果 point_tensor 中有多个点，则将其依次显示
            points = point_tensor.numpy()
            # 这里 points 的第一维对应 mask_tensor 的通道数，
            # 如果与通道数不匹配，也可做相应检查
            # 先简单画所有点（或按需要修改）
            if len(points.shape) == 2:
                for i in range(points.shape[0]):
                    x, y = points[i]
                    axs[0].scatter(x, y, c='red', s=40)

            # 其余子图：依次显示各通道的 mask
            for c in range(n_channels):
                axs[c + 1].imshow(mask_tensor[c].numpy(), cmap='gray')
                axs[c + 1].set_title(f"Mask channel={c}")
                axs[c + 1].axis("off")

            plt.suptitle(f"Path: {img_path}")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    test_all_modes()