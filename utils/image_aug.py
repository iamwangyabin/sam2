"""
Image augmentation utilities using Albumentations.

This module provides functions to generate random image augmentation 
transforms with the albumentations library. It includes functions to create 
individual transforms, compose a sequence of transforms, and apply additional 
post-processing (e.g., Gaussian blur, noise, JPEG compression, and gamma correction).
"""

import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import os
import cv2
from tqdm import tqdm
import random
import albumentations as A

def make_single_albu_post(key: str) -> A.BasicTransform:
    """
    根据传入的 key 创建一个 Albumentations 转换对象。

    参数:
        key (str): 表示转换类型的关键字，支持的 key 包括：
                   'Resize', 'Blur', 'GaussianBlur', 'MedianBlur', 'MotionBlur', 
                   'Downscale', 'GaussNoise', 'ISONoise', 'RandomBrightnessContrast',
                   'RandomGamma', 'CLAHE', 'HueSaturationValue'。

    返回:
        配置了随机参数的 Albumentations 转换实例。

    异常:
        ValueError: 当 key 不被支持时抛出。
    """
    if key == 'Resize':
        height_ratio = random.choice([np.random.uniform(0.5, 0.9), np.random.uniform(1.1, 1.5)])
        width_ratio = random.choice([np.random.uniform(0.5, 0.9), np.random.uniform(1.1, 1.5)])
        height = int(height_ratio * 2048)
        width = int(width_ratio * 2048)
        tf_str = f"A.{key}(height={height}, width={width}, p=1, interpolation=1)"
    elif 'Blur' in key:
        blur_limit = random.choice([3, 5, 7])
        tf_str = f"A.{key}(blur_limit=({blur_limit}, {blur_limit}), p=1)"
    elif key == 'Downscale':
        scale_min = random.uniform(0.7, 1)
        scale_max = scale_min
        tf_str = f"A.{key}(scale_min={scale_min}, scale_max={scale_max}, p=1, interpolation=1)"
    elif key == 'GaussNoise':
        var_limit = random.uniform(10, 40)
        tf_str = f"A.{key}(var_limit=({var_limit}, {var_limit}), mean=0, p=1)"
    elif key == 'ISONoise':
        color_shift = random.uniform(0.01, 0.04)
        intensity = random.uniform(0.1, 0.5)
        tf_str = f"A.{key}(color_shift=({color_shift}, {color_shift}), intensity=({intensity}, {intensity}), p=1)"
    elif key == 'RandomBrightnessContrast':
        brightness_limit = random.uniform(-0.1, 0.1)
        contrast_limit = random.uniform(-0.1, 0.1)
        # 确保亮度和对比度的变化足够明显（避免都接近零）
        while abs(brightness_limit) < 0.05 and abs(contrast_limit) < 0.05:
            brightness_limit = random.uniform(-0.1, 0.1)
            contrast_limit = random.uniform(-0.1, 0.1)
        tf_str = f"A.{key}(brightness_limit=({brightness_limit}, {brightness_limit}), contrast_limit=({contrast_limit}, {contrast_limit}), p=1)"
    elif key == 'RandomGamma':
        gamma_limit = int(random.uniform(60, 150))
        # 保证 gamma 参数和100的差距足够大
        while abs(gamma_limit - 100) < 10:
            gamma_limit = int(random.uniform(60, 150))
        tf_str = f"A.{key}(gamma_limit=({gamma_limit}, {gamma_limit}), p=1)"
    elif key == 'CLAHE':
        clip_limit = random.uniform(1, 4)
        tile_grid_size = 2 ** random.choice(np.arange(1, 4))
        tf_str = f"A.{key}(clip_limit=({clip_limit}, {clip_limit}), tile_grid_size=({tile_grid_size}, {tile_grid_size}), p=1)"
    elif key == 'HueSaturationValue':
        hue_shift_limit = random.uniform(-10, 10)
        sat_shift_limit = random.uniform(-15, 15)
        val_shift_limit = random.uniform(-10, 10)
        # 确保色相、饱和度和亮度变化足够明显
        while abs(hue_shift_limit) < 2.5 and abs(sat_shift_limit) < 2.5 and abs(val_shift_limit) < 2.5:
            hue_shift_limit = random.uniform(-10, 10)
            sat_shift_limit = random.uniform(-15, 15)
            val_shift_limit = random.uniform(-10, 10)
        tf_str = f"A.{key}(hue_shift_limit=({hue_shift_limit}, {hue_shift_limit}), sat_shift_limit=({sat_shift_limit}, {sat_shift_limit}), val_shift_limit=({val_shift_limit}, {val_shift_limit}), p=1)"
    else:
        raise ValueError(f"Unsupported key '{key}' for augmentation.")

    # 根据拼接的字符串创建 Albumentations 转换实例
    return eval(tf_str)


# Grouped keys that represent different categories of transformations,
# along with它们的采样权重（用于后处理时随机选取变换种类）。
KEY_LIST = [
    ['Resize'],
    ['Blur', 'GaussianBlur', 'MedianBlur', 'MotionBlur'],
    ['Downscale'],
    ['GaussNoise', 'ISONoise'],
    ['RandomBrightnessContrast'],
    ['RandomGamma'],
    ['CLAHE'],
    ['HueSaturationValue']
]
KEY_WEIGHTS = [2, 1, 1, 1, 1, 1, 1, 1]


def weighted_sample_without_replacement(population: list, weights: list, k: int) -> list:
    """
    根据给定的权重从 population 中不放回地随机抽取 k 个样本。

    参数:
        population (list): 每个元素为一个键列表（例如支持的转换种类）。
        weights (list): 每个 group 对应的采样权重。
        k (int): 需要抽取的样本数量。

    返回:
        抽样得到的键列表，每个键来自对应的 group。
    """
    chosen = []
    for _ in range(k):
        cumulative_weights = [sum(weights[:i + 1]) for i in range(len(weights))]
        total = cumulative_weights[-1]
        r = random.uniform(0, total)
        for i, cw in enumerate(cumulative_weights):
            if cw >= r:
                chosen.append(random.choice(population.pop(i)))
                weights.pop(i)
                break
    return chosen


def make_albu_post(num: int = 1) -> A.Compose:
    """
    创建一个 Albumentations Compose 对象，该对象包含一系列随机选取的转换，
    最后总是附加一个 1024x1024 的 Resize 转换。

    参数:
        num (int): 除最后的 Resize 之外随机选取的转换数量。默认为 1。

    返回:
        Albumentations Compose 对象。
    """
    keys = weighted_sample_without_replacement(population=KEY_LIST.copy(), weights=KEY_WEIGHTS.copy(), k=num)
    albu_list = [make_single_albu_post(key) for key in keys]
    # 加入最后统一 resize 到 1024x1024
    albu_list.append(A.Resize(height=1024, width=1024, p=1, interpolation=1))
    return A.Compose(albu_list)


def post_processing(fore_image: np.ndarray, fore_mask: np.ndarray) -> tuple:
    """
    对前景图像和对应的掩码应用随机后处理增强操作。

    参数:
        fore_image (np.ndarray): 前景图像数组。
        fore_mask (np.ndarray): 前景掩码数组。

    返回:
        (augmented_image, augmented_mask) 增强后的图像和掩码。
    """
    num_fore = random.choices([1, 2, 3, 4], weights=[4, 3, 2, 1], k=1)[0]
    albu_post_fore = make_albu_post(num=num_fore)
    albu_result_fore = albu_post_fore(image=fore_image, mask=fore_mask)
    return albu_result_fore['image'], albu_result_fore['mask']


def Gaussian_Blur(img_RGB: np.ndarray, mask: np.ndarray, kernel_size: int) -> tuple:
    """
    对输入图像应用高斯模糊，并返回模糊后的图像及原掩码。

    参数:
        img_RGB (np.ndarray): 输入图像数组。
        mask (np.ndarray): 对应的掩码数组。
        kernel_size (int): 高斯模糊核尺寸（通常为奇数）。

    返回:
        (blurred_image, mask) 经过模糊处理的图像和原始掩码。
    """
    post_process = A.Compose([
        A.GaussianBlur(blur_limit=(kernel_size, kernel_size), always_apply=True),
    ])
    aug = post_process(image=img_RGB, mask=mask)
    return aug['image'], aug['mask']


def Gaussian_Noise(img_RGB: np.ndarray, stddev: float) -> np.ndarray:
    """
    对输入图像添加高斯噪声。

    参数:
        img_RGB (np.ndarray): 输入图像数组。
        stddev (float): 噪声的标准差。

    返回:
        含有高斯噪声的图像数组。
    """
    mean = 0
    noise = np.random.normal(mean, stddev, img_RGB.shape)
    noisy_img = np.clip(img_RGB.astype(np.float32) + noise.astype(np.float32), 0, 255).astype(np.uint8)
    return noisy_img


# 预先生成用于 JPEG 压缩的转换字典，其 key 为质量因子。
comp_dict = {q: A.ImageCompression(quality_lower=q-1, quality_upper=q, p=1) for q in range(50, 101)}


def JPEG_Compress(img_RGB: np.ndarray, QF: int) -> np.ndarray:
    """
    对输入图像应用 JPEG 压缩。

    参数:
        img_RGB (np.ndarray): 输入图像数组。
        QF (int): JPEG 压缩的质量因子（应在50到100之间）。

    返回:
        JPEG 压缩后的图像数组。
    """
    comp_fn = comp_dict[QF]
    return comp_fn(image=img_RGB)['image']


def Gamma_Correction(img_RGB: np.ndarray, gamma: float) -> np.ndarray:
    """
    对输入图像应用伽马校正。

    参数:
        img_RGB (np.ndarray): 输入图像数组。
        gamma (float): 伽马校正参数。

    返回:
        伽马校正后的图像数组。
    """
    corrected = (img_RGB / 255.0) ** (1.0 / gamma) * 255.0
    return corrected.astype(np.uint8)


def robust_post_processing(img: np.ndarray, mask: np.ndarray, pp_type: str, pp_param) -> tuple:
    """
    根据指定的类型和参数，对图像及其掩码应用稳健的后处理操作。

    参数:
        img (np.ndarray): 输入图像数组。
        mask (np.ndarray): 对应的掩码数组。
        pp_type (str or None): 后处理类型，支持：
                               'Gaussian_Blur', 'Gaussian_Noise', 
                               'JPEG_Compress', 'Gamma_Correction'。
                               若为 None，则不作处理。
        pp_param: 针对不同后处理类型的参数。例如，高斯模糊的核尺寸、噪声标准差、
                  JPEG 压缩的质量因子或伽马校正的伽马值。

    返回:
        (processed_image, processed_mask) 处理后的图像和掩码。

    异常:
        ValueError: 当 pp_type 或 pp_param 不符合要求时抛出异常。
    """
    if pp_type is None:
        return img, mask
    if not isinstance(pp_type, str):
        raise ValueError("pp_type must be a string or None.")

    if pp_type == 'Gaussian_Blur':
        if pp_param not in range(3, 20, 4):
            raise ValueError("pp_param for Gaussian_Blur must be in range(3, 20, 4).")
        ret_img, ret_mask = Gaussian_Blur(img, mask, kernel_size=pp_param)
    elif pp_type == 'Gaussian_Noise':
        if pp_param not in range(3, 24, 4):
            raise ValueError("pp_param for Gaussian_Noise must be in range(3, 24, 4).")
        ret_img = Gaussian_Noise(img, stddev=pp_param)
        ret_mask = mask
    elif pp_type == 'JPEG_Compress':
        if pp_param not in range(50, 101, 10):
            raise ValueError("pp_param for JPEG_Compress must be in range(50, 101, 10).")
        ret_img = JPEG_Compress(img, QF=pp_param)
        ret_mask = mask
    elif pp_type == 'Gamma_Correction':
        valid_gamma = [0.7 + 0.15 * t for t in range(5)]
        if pp_param not in valid_gamma:
            raise ValueError(f"pp_param for Gamma_Correction must be one of {valid_gamma}.")
        ret_img = Gamma_Correction(img, gamma=pp_param)
        ret_mask = mask
    else:
        raise ValueError(f"Unsupported pp_type: {pp_type}.")

    return ret_img, ret_mask