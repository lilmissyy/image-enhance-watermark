import cv2
import numpy as np


def histogram_equalization(img_path):
    """
    简化版直方图均衡化（课程设计核心图像处理算法）
    功能：提升图像对比度，逻辑简化，无冗余操作
    输入：图像路径，输出：增强后BGR数组
    """
    # 读取图像（仅支持JPG/PNG，简化格式判断）
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像：{img_path}")

    # 简化处理逻辑：彩色图转灰度均衡后转回，灰度图直接均衡
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized_gray = cv2.equalizeHist(gray)
        equalized_img = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)
    else:
        equalized_img = cv2.equalizeHist(img)

    return equalized_img