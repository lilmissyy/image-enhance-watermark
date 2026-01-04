import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1_path, img2_path):
    """简化PSNR计算（课程设计实验分析指标）"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError("图像路径错误")

    # 简化尺寸统一：直接resize，不复杂判断
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')


def calculate_ssim(img1_path, img2_path):
    """简化SSIM计算（课程设计实验分析指标）"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("图像路径错误")

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    return ssim(img1, img2, win_size=11, data_range=255)