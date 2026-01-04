import numpy as np
import pywt
from PIL import Image
import cv2

# 简化配置（通用适配，无硬编码）
ALPHA = 0.1  # 信号强度（简化为固定值，确保提取）
FIXED_THRESHOLD = 0.04  # 固定阈值，避免动态计算复杂度
MIN_IMAGE_SIZE = 150  # 降低最小尺寸要求，简化适配
ALLOWED_CHARS = set(chr(i) for i in range(32, 127))  # 支持所有ASCII字符


def embed_dwt_watermark(enhanced_img_input, watermark_text, output_path, is_path=True):
    """
    简化版DWT水印嵌入（通用版，支持任意ASCII水印）
    核心简化：1级DWT、固定嵌入位置、PNG无损保存
    """
    # 1. 简化水印校验（仅校验ASCII，无长度硬编码）
    watermark_text = watermark_text.strip()
    if not watermark_text:
        raise ValueError("水印文本不能为空")
    for char in watermark_text:
        if char not in ALLOWED_CHARS:
            raise ValueError(f"仅支持ASCII字符，不支持'{char}'")

    # 2. 水印转二进制（简化转换逻辑）
    watermark_bin = ''.join([format(ord(c), '08b') for c in watermark_text])
    watermark_len = len(watermark_bin)

    # 3. 图像读取（简化通道处理，强制RGB去透明）
    if is_path:
        img = Image.open(enhanced_img_input).convert('RGB').convert('YCbCr')
    else:
        rgb_img = cv2.cvtColor(enhanced_img_input, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_img).convert('YCbCr')

    y_channel, cr, cb = img.split()
    y_array = np.array(y_channel, dtype=np.float32)
    h, w = y_array.shape

    # 4. 简化尺寸校验（仅校验最小尺寸）
    if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
        raise ValueError(f"图像需≥{MIN_IMAGE_SIZE}×{MIN_IMAGE_SIZE}像素")

    # 5. 尺寸补全（简化为4的整数倍，确保DWT无错）
    def pad_to_multiple(arr, multiple=4):
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        padded = np.zeros((new_h, new_w), dtype=arr.dtype)
        padded[:h, :w] = arr
        return padded

    y_padded = pad_to_multiple(y_array)
    # 简化DWT：仅1级变换，取低频分量cA1
    cA1, _ = pywt.dwt2(y_padded, 'haar')

    # 6. 简化嵌入逻辑：行优先遍历，填满水印为止
    idx = 0
    for i in range(cA1.shape[0]):
        for j in range(cA1.shape[1]):
            if idx < watermark_len:
                cA1[i][j] += ALPHA * int(watermark_bin[idx])
                idx += 1
            else:
                break
        if idx >= watermark_len:
            break

    # 7. 逆变换+简化保存（强制PNG无损）
    new_coeffs = (cA1, pywt.dwt2(y_padded, 'haar')[1])
    new_y = pywt.idwt2(new_coeffs, 'haar')[:h, :w]
    new_y = np.clip(new_y, 0, 255).astype(np.uint8)

    new_img = Image.merge('YCbCr', (Image.fromarray(new_y), cr, cb)).convert('RGB')
    new_img.save(output_path, format='PNG')


def extract_dwt_watermark(watermarked_img_path, watermark_len):
    """
    简化版DWT水印提取（通用版，输入长度即可）
    核心简化：固定阈值、对应嵌入位置提取
    """
    if watermark_len <= 0:
        raise ValueError("水印长度必须为正整数")
    target_bits = watermark_len * 8

    # 1. 图像读取（与嵌入逻辑一致）
    img = Image.open(watermarked_img_path).convert('RGB').convert('YCbCr')
    y_channel, _, _ = img.split()
    y_array = np.array(y_channel, dtype=np.float32)
    h, w = y_array.shape

    # 2. 尺寸补全（简化复用函数）
    def pad_to_multiple(arr, multiple=4):
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        padded = np.zeros((new_h, new_w), dtype=arr.dtype)
        padded[:h, :w] = arr
        return padded

    y_padded = pad_to_multiple(y_array)
    cA1, _ = pywt.dwt2(y_padded, 'haar')

    # 3. 简化提取逻辑：对应嵌入位置，固定阈值判断
    watermark_bin = ''
    idx = 0
    for i in range(cA1.shape[0]):
        for j in range(cA1.shape[1]):
            if idx < target_bits:
                decimal = cA1[i][j] - np.floor(cA1[i][j])
                watermark_bin += '1' if decimal > FIXED_THRESHOLD else '0'
                idx += 1
            else:
                break
        if idx >= target_bits:
            break

    # 4. 二进制转文本（简化分组逻辑）
    watermark_text = ''
    for i in range(0, target_bits, 8):
        byte = watermark_bin[i:i + 8]
        if len(byte) == 8:
            watermark_text += chr(int(byte, 2))

    return watermark_text