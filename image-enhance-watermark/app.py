from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2

# 导入简化后的算法模块
from algorithm.image_enhance import histogram_equalization
from algorithm.image_watermark import embed_dwt_watermark, extract_dwt_watermark
from algorithm.image_utils import calculate_psnr, calculate_ssim

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# 配置路径（简化，直接拼接）
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, 'static', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(APP_ROOT, 'static', 'processed')

# 自动创建文件夹（简化逻辑，无复杂判断）
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 简化文件上传验证
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '' or not file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
            return redirect(request.url)

        # 保存原图（简化命名，直接用原文件名）
        img_filename = file.filename
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        file.save(original_path)

        try:
            # 简化尺寸校验
            img = cv2.imread(original_path)
            h, w = img.shape[:2]
            if h < 150 or w < 150:
                return f"图像需≥150×150像素<br><a href='/'>返回</a>"

            # 核心流程：增强→嵌入水印（简化参数传递）
            enhanced_img = histogram_equalization(original_path)
            # 获取用户输入的水印和长度（通用，无硬编码）
            watermark_text = request.form.get('watermark', 'test123456').strip()
            watermark_len = len(watermark_text)

            # 生成结果图（简化命名，加png后缀）
            final_filename = f'result_{os.path.splitext(img_filename)[0]}.png'
            final_path = os.path.join(app.config['PROCESSED_FOLDER'], final_filename)
            embed_dwt_watermark(enhanced_img, watermark_text, final_path, is_path=False)

            # 提取验证+质量指标（简化，直接调用）
            extracted_wm = extract_dwt_watermark(final_path, watermark_len)
            psnr_val = calculate_psnr(original_path, final_path)
            ssim_val = calculate_ssim(original_path, final_path)

            # 传递结果到前端（简化参数，仅核心数据）
            return render_template(
                'index.html',
                original_img=img_filename,
                final_img=final_filename,
                watermark_text=watermark_text,
                watermark_len=watermark_len,
                extracted_wm=extracted_wm,
                psnr=round(psnr_val, 2),
                ssim=round(ssim_val, 4)
            )
        except Exception as e:
            return f"执行失败：{str(e)}<br><a href='/'>返回</a>"

    # GET请求返回前端（简化，无额外逻辑）
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract_route():
    """简化提取接口，仅处理核心逻辑"""
    if 'watermarked_image' not in request.files:
        return jsonify({"status": "error", "msg": "请上传PNG结果图"})

    file = request.files['watermarked_image']
    if file.filename == '' or not file.filename.lower().endswith('png'):
        return jsonify({"status": "error", "msg": "仅支持PNG格式"})

    # 获取用户输入的水印长度
    watermark_len = request.form.get('watermark_length', '').strip()
    if not watermark_len.isdigit() or int(watermark_len) <= 0:
        return jsonify({"status": "error", "msg": "水印长度必须为正整数"})
    watermark_len = int(watermark_len)

    # 临时保存（简化命名）
    temp_path = os.path.join(app.config['PROCESSED_FOLDER'], 'temp_extract.png')
    file.save(temp_path)

    try:
        extracted_wm = extract_dwt_watermark(temp_path, watermark_len)
        os.remove(temp_path)
        return jsonify({
            "status": "success",
            "extracted_watermark": extracted_wm
        })
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"status": "error", "msg": str(e)})


# 简化图像访问路由
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


# 程序入口（简化，固定端口）
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)