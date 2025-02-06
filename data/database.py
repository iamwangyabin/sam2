import os
import random
import cv2
import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms

from utils.image_aug import post_processing, robust_post_processing


class FileDataset(Dataset):
    def __init__(self, root_path, im_list_file):
        self.root_path = root_path
        with open(im_list_file, 'r') as f:
            self.im_list = [line.strip().split(',') for line in f if line.strip()]

    def __len__(self):
        return len(self.im_list)

    def get_img_path(self, index):
        return os.path.join(self.root_path, self.im_list[index][0])

    def get_mask_path(self, index):
        mask_str = self.im_list[index][1].strip()
        return os.path.join(self.root_path, mask_str) if mask_str != "None" else None

    def get_img(self, index):
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(self.get_img_path(index)).convert('RGB')
        return img

    def get_mask(self, index):
        mask_path = self.get_mask_path(index)
        if mask_path is None:
            return None
        elif str(mask_path).endswith('.npz'):
            mask = np.load(mask_path)['arr_0'].squeeze()
            mask[mask > 0] = 1
            return mask

        else:
            mask_img = Image.open(mask_path).convert("L")
            mask = np.array(mask_img)
            mask[mask > 0] = 1
            return mask


    def shuffle_im_list(self, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(self.im_list)


def dilate_mask(mask, dilate_factor=3):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


class CoreDataset(Dataset):
    def __init__(
        self, 
        datasets_list, 
        mode='train', 
        imsize=1024, 
        augment_type=0, 
        num_pairs=1, 
        pp_type=None, 
        pp_param=None, 
        resize_mode=None, 
        crop_prob=0
    ):
        """
        :param datasets_list: list[FileDataset or AbstractForgeryDataset], 多个数据集的列表
        :param mode: 'train', 'valid', 'test'等，不同mode下图像和mask的处理流程不一样
        :param imsize: 目标图像尺寸(默认1024)
        :param augment_type: 图像增强类型(0表示不增强, 1表示轻度增强, 2表示更强后处理)
        :param num_pairs: 当mode='pair_random'时需要采样的点对数量
        :param pp_type: 后处理类型
        :param pp_param: 后处理参数
        :param resize_mode: 可选[None, 'crop_prob', 'resize_when_large']
        :param crop_prob: 执行随机裁剪的概率
        """
        assert len(datasets_list) >= 1, "Must not be an empty list"
        self.datasets_list = datasets_list
        # 累积长度前缀，用于根据idx锁定具体属于哪个dataset
        self.dataset_start_indices = [
            sum([len(ds) for ds in dss]) 
            for dss in (datasets_list[:i] for i in range(len(datasets_list)))
        ]
        self.mode = mode
        self.imsize = imsize
        self.augment_type = augment_type
        self.num_pairs = num_pairs
        self.pp_type = pp_type
        self.pp_param = pp_param
        assert resize_mode in [None, 'crop_prob', 'resize_when_large'], f'Invalid resize_mode: {resize_mode}'
        self.resize_mode = resize_mode
        self.crop_prob = crop_prob

    def __len__(self):
        return self.dataset_start_indices[-1] + len(self.datasets_list[-1])

    def __getitem__(self, idx):
        dataset_idx = self._locate_dataset_index(idx)
        img_idx = idx - self.dataset_start_indices[dataset_idx]

        img, mask, mask_padinfo, img_path = self.load_item(dataset_idx, img_idx)

        if self.mode in ["test_auto"]:
            return self._handle_test_auto(img, mask, mask_padinfo, img_path)
        elif self.mode in ["im_mask"]:
            return self._handle_im_mask(img, mask, img_path)
        elif self.mode in ["train", "valid", "one_random"]:
            return self._handle_single_point_mode(img, mask, mask_padinfo, img_path)
        elif self.mode in ["mid", "one_mid", "complete_random"]:
            return self._handle_mid_or_complete_random(img, mask, mask_padinfo, img_path)
        elif self.mode in ["pair_random", "pair_random_cc"]:
            return self._handle_pair_random(img, mask, mask_padinfo, img_path)
        else:
            raise ValueError(f"Not supported value self.mode: {self.mode}.")

    def load_item(self, dataset_idx, img_idx):

        dataset = self.datasets_list[dataset_idx]
        img_pil = dataset.get_img(img_idx)  # PIL
        W, H = img_pil.size
        img = np.array(img_pil)  # shape=(H,W,3), dtype=uint8
        mask = dataset.get_mask(img_idx)

        if mask is None:
            mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
        else:
            mask = mask.astype(np.uint8)
        img, mask = self._apply_augmentations(img, mask)
        img, mask, mask_padinfo = self._resize_or_crop(img, mask, W, H)
        return img, mask, mask_padinfo, dataset.get_img_path(img_idx)


    def _locate_dataset_index(self, global_idx):
        return max(
            (i for i, x in enumerate(self.dataset_start_indices) if x <= global_idx),
            default=0
        )

    def _apply_augmentations(self, img, mask):
        if self.augment_type == 1 and np.random.rand() < 0.5:
            img, mask = post_processing(img, mask)
        elif self.augment_type == 2:
            img, mask = robust_post_processing(img, mask, self.pp_type, self.pp_param)
        return img, mask

    def _resize_or_crop(self, img, mask, orig_w, orig_h):
        mask_padinfo = None
        if (self.resize_mode == 'crop_prob' 
            and self.crop_prob > 0 
            and np.random.rand() < self.crop_prob):

            if orig_h < self.imsize or orig_w < self.imsize:
                pad_h = max(0, self.imsize - orig_h)
                pad_w = max(0, self.imsize - orig_w)
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                             mode='constant', constant_values=0)
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
                
                transform = A.Compose([
                    A.RandomCrop(width=self.imsize, height=self.imsize),
                ])
                transformed = transform(image=img, mask=mask)
                img, mask = transformed['image'], transformed['mask']

                mask_padinfo = np.ones_like(mask)
                mask_padinfo[orig_h:, :] = 0
                mask_padinfo[:, orig_w:] = 0
            else:
                transform = A.Compose([
                    A.RandomCrop(width=self.imsize, height=self.imsize),
                ])
                transformed = transform(image=img, mask=mask)
                img, mask = transformed['image'], transformed['mask']

        elif self.resize_mode == 'resize_when_large' and max(orig_h, orig_w) < self.imsize:
            pad_h = max(0, self.imsize - orig_h)
            pad_w = max(0, self.imsize - orig_w)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)),
                         mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask_padinfo = np.ones_like(mask)
            mask_padinfo[orig_h:, :] = 0
            mask_padinfo[:, orig_w:] = 0

        else:
            img = cv2.resize(img, (self.imsize, self.imsize))
            mask = cv2.resize(mask, (self.imsize, self.imsize), interpolation=cv2.INTER_NEAREST)

        return img, mask, mask_padinfo

    def _handle_test_auto(self, img, mask, mask_padinfo, img_path):
        img = img.astype('float32')
        if mask_padinfo is not None:
            mask[mask_padinfo == 0] = -1
            img[mask_padinfo == 0] = 0
        mask_tensor = torch.tensor(mask[None, ...], dtype=torch.int64)
        return img, mask_tensor, str(img_path)

    def _handle_im_mask(self, img, mask, img_path):
        img = img.transpose((2, 0, 1)).astype('float32')
        img_tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.int64)
        return img_tensor, mask_tensor, str(img_path)

    def _handle_single_point_mode(self, img, mask, mask_padinfo, img_path):
        target_value = np.random.choice(np.unique(mask))
        y_indices, x_indices = (
            np.where((mask == target_value) & (mask_padinfo == 1))
            if mask_padinfo is not None 
            else np.where(mask == target_value)
        )
        index = np.random.randint(len(x_indices))
        point = np.array([x_indices[index], y_indices[index]])

        if target_value == 0:
            mask = 1 - mask

        mask = mask[None, ...]
        point = point[None, ...]

        if mask_padinfo is not None:
            mask[np.repeat(mask_padinfo[None, ...], mask.shape[0], axis=0) == 0] = -1
            img[mask_padinfo == 0] = 0

        img_tensor, mask_tensor, point_tensor = self._build_tensor(img, mask, point)
        return img_tensor, mask_tensor, point_tensor, str(img_path)

    def _handle_mid_or_complete_random(self, img, mask, mask_padinfo, img_path):
        if self.mode in ["mid", "one_mid"]:
            if mask_padinfo is not None:
                y_indices, x_indices = np.where(mask_padinfo == 1)
                index = len(x_indices) // 2
                point = np.array([x_indices[index], y_indices[index]])
            else:
                point = np.array([self.imsize // 2, self.imsize // 2])
        else:
            # complete_random
            y_indices, x_indices = (
                np.where(mask_padinfo == 1) 
                if mask_padinfo is not None 
                else np.where(mask == mask)
            )
            index = np.random.randint(len(x_indices))
            point = np.array([x_indices[index], y_indices[index]])

        mask = mask[None, ...]
        point = point[None, ...]

        if mask_padinfo is not None:
            mask[np.repeat(mask_padinfo[None, ...], mask.shape[0], axis=0) == 0] = -1
            img[mask_padinfo == 0] = 0

        img_tensor, mask_tensor, point_tensor = self._build_tensor(img, mask, point)
        return img_tensor, mask_tensor, point_tensor, str(img_path)

    def _handle_pair_random(self, img, mask, mask_padinfo, img_path):

        unique_list = np.unique(mask)
        points, masks = [], []
        for _ in range(self.num_pairs):
            if len(unique_list) == 2:
                point0, mask0 = self._sample_point_and_mask(mask, 0, mask_padinfo)
                points.append(point0)
                masks.append(mask0)

                point1, mask1 = self._sample_point_and_mask(mask, 1, mask_padinfo)
                points.append(point1)
                masks.append(mask1)
            else:
                v = unique_list[0]
                point0, mask0 = self._sample_point_and_mask(mask, v, mask_padinfo)
                points.append(point0)
                masks.append(1 - mask if v == 0 else mask)

                point1, mask1 = self._sample_point_and_mask(mask, v, mask_padinfo)
                points.append(point1)
                masks.append(1 - mask if v == 0 else mask)

        if self.mode == "pair_random_cc":
            for i in range(len(masks)):
                new_mask = self._apply_connected_component(masks[i], points[i])
                masks[i] = new_mask

        point = np.stack(points, axis=0)
        mask_batched = np.stack(masks, axis=0)

        if mask_padinfo is not None:
            mask_batched[
                np.repeat(mask_padinfo[None, ...], mask_batched.shape[0], axis=0) == 0
            ] = -1
            img[mask_padinfo == 0] = 0

        img_tensor, mask_tensor, point_tensor = self._build_tensor(img, mask_batched, point)
        return img_tensor, mask_tensor, point_tensor, str(img_path)

    def _sample_point_and_mask(self, mask, val, mask_padinfo):

        if mask_padinfo is not None:
            y_indices, x_indices = np.where((mask == val) & (mask_padinfo == 1))
        else:
            y_indices, x_indices = np.where(mask == val)

        idx_rand = np.random.randint(len(x_indices))
        point = np.array([x_indices[idx_rand], y_indices[idx_rand]])
        # 对于0返回1-mask, 对于1返回 mask即可(为对应类别)
        return point, (1 - mask if val == 0 else mask)

    def _apply_connected_component(self, mask_val, point):

        cnt1, labels1 = cv2.connectedComponents(mask_val, connectivity=4)
        cnt2, labels2 = cv2.connectedComponents(255 - mask_val, connectivity=4)
        label3 = labels1 + labels2 * cnt1
        point_value_mask = (label3 == label3[point[1], point[0]]).astype(np.uint8)
        dilated_mask = dilate_mask(point_value_mask)
        new_mask = np.array(mask_val, copy=True, dtype=int)
        new_mask[np.logical_not(np.isin(label3, label3[dilated_mask == 1]))] = -1
        return new_mask

    def _build_tensor(self, img, mask, point):

        img = img.transpose((2, 0, 1)).astype('float32')  # (H,W,3)->(3,H,W)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.int64)
        point_tensor = torch.tensor(point, dtype=torch.float32)
        return img_tensor, mask_tensor, point_tensor


    def shuffle_im_lists(self, random_seed=None):
        for ds in self.datasets_list:
            ds.shuffle_im_list(random_seed)