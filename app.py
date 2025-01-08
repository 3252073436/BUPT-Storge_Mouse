import os
import shutil
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from PIL import Image
import torch
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
import numpy as np
import subprocess

app = Flask(__name__)
app.secret_key = "secret_key"

# GPU or CPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 在model_path这里填上你电脑里的OFA-Syschinese-clip-vit-base-patch16（或其他模型）文件夹所在路径
model_path = r"F:\仓鼠2式\仓鼠2式\models\OFA-Syschinese-clip-vit-base-patch16"
model = ChineseCLIPModel.from_pretrained(model_path).to(device)
processor = ChineseCLIPProcessor.from_pretrained(model_path)
# Database setup
db_path = "features.db"
current_folder = None  # Global variable for current folder
scan_progress = 0  # Global variable for progress tracking

# Path to ffmpeg
ffmpeg_path = os.path.join(os.getcwd(), 'ffmpeg', 'bin', 'ffmpeg.exe')

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS image_features')
    cursor.execute('''CREATE TABLE image_features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feature BLOB,
                        file_path TEXT UNIQUE,
                        original_file_path TEXT UNIQUE  -- 新增原始文件路径列
                      )''')
    conn.commit()
    conn.close()



init_db()

# Feature extraction
def extract_image_features(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()

def extract_video_frame(video_path):
    # Use ffmpeg to extract a frame from the video (not the first and last frames)
    output_image_path = video_path + "_frame.jpg"
    cmd = [
        ffmpeg_path, '-i', video_path, '-vf', 'select=eq(n\,30)', '-vsync', 'vfr', output_image_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_image_path

def copy_to_static(file_path):
    static_dir = os.path.join(os.getcwd(), 'static', 'images')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    destination = os.path.join(static_dir, os.path.basename(file_path))
    if not os.path.exists(destination):
        shutil.copy(file_path, destination)

    return os.path.join('images', os.path.basename(file_path)).replace("\\", "/")






def clear_static_images():
    static_dir = os.path.join(os.getcwd(), 'static', 'images')
    if os.path.exists(static_dir):
        for file_name in os.listdir(static_dir):
            file_path = os.path.join(static_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)


def clear_database():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 清理数据库中的所有数据
    cursor.execute('DELETE FROM image_features')
    conn.commit()  # 提交删除操作

    # 执行 VACUUM 操作，回收数据库空间
    cursor.execute('VACUUM')
    conn.commit()  # 提交 VACUUM 操作

    conn.close()

def scan_and_extract_features(folder_path):
    global scan_progress
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 清理数据库中的所有数据
    cursor.execute('DELETE FROM image_features')
    conn.commit()  # 提交清理操作

    all_files = [f for r, d, files in os.walk(folder_path) for f in files]
    total_files = len(all_files)
    processed_files = 0

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
                try:
                    features = extract_image_features(file_path)
                    static_path = copy_to_static(file_path)  # 复制图片到 static 目录
                    cursor.execute(
                        'INSERT OR IGNORE INTO image_features (feature, file_path, original_file_path) VALUES (?, ?, ?)',
                        (features.tobytes(), static_path, file_path))  # 保存相对路径与绝对路径
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            elif file_name.lower().endswith(('mp4', 'avi', 'mov')):
                try:
                    frame_path = extract_video_frame(file_path)
                    features = extract_image_features(frame_path)
                    static_path = copy_to_static(frame_path)
                    cursor.execute(
                        'INSERT OR IGNORE INTO image_features (feature, file_path, original_file_path) VALUES (?, ?, ?)',
                        (features.tobytes(), static_path, file_path))
                    os.remove(frame_path)  # 可选：删除提取的帧
                except Exception as e:
                    print(f"Error processing video {file_path}: {e}")
            processed_files += 1
            scan_progress = (processed_files / total_files) * 100

    conn.commit()
    conn.close()



def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def query_by_text(query_text, k=None, similarity_threshold=None):
    inputs = processor(text=query_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs).cpu().numpy().flatten()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, feature, file_path FROM image_features')
    rows = cursor.fetchall()

    results = []
    for row in rows:
        feature = np.frombuffer(row[1], dtype=np.float32)
        similarity = cosine_similarity(text_features, feature)
        if similarity_threshold is None or similarity >= similarity_threshold:
            static_path = row[2]  # 相对路径（存储在数据库中的）
            # 绝对路径是文件最初所在的原始文件夹路径
            original_path = os.path.join(current_folder, static_path.replace('images/', ''))  # 还原为原始路径
            results.append((static_path, similarity, original_path))  # 返回原始路径

    conn.close()
    results.sort(key=lambda x: x[1], reverse=True)

    if k:
        results = results[:k]

    return results


@app.route('/scan', methods=['POST'])
def scan():
    global current_folder, scan_progress
    folder_path = request.form['folder_path']

    # 检查是否是新的文件夹路径
    if folder_path != current_folder:
        # 如果是新文件夹，清理数据库
        clear_database()
        scan_progress = 0  # Reset progress
        clear_static_images()
        scan_and_extract_features(folder_path)
        if folder_path != current_folder:
            flash(f"成功加载新文件夹: {folder_path}")
        current_folder = folder_path
    else:
        flash("您已经扫描过该文件夹，不需要清理数据库。")

    return redirect(url_for('index'))


@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify({"progress": scan_progress})


@app.route('/')
def index():
    return render_template('index.html', similar_images=None, current_folder=current_folder)


@app.route('/query', methods=['POST'])
def query():
    query_text = request.form['query_text']
    top_k = request.form.get('top_k', '').strip()
    similarity_threshold = request.form.get('similarity_threshold', '').strip()

    k = int(top_k) if top_k.isdigit() else None
    similarity_threshold = float(similarity_threshold) if similarity_threshold else None

    if not query_text:
        flash("请先输入目标词条")
        return redirect(url_for('index'))

    similar_file_paths = query_by_text(query_text, k=k, similarity_threshold=similarity_threshold)
    if not similar_file_paths:
        flash("没有符合条件的结果")
        return redirect(url_for('index'))

    return render_template('results.html', similar_images=similar_file_paths)


@app.route('/clear_images', methods=['POST'])
def clear_images():
    clear_static_images()
    flash("Images cleared.")
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
