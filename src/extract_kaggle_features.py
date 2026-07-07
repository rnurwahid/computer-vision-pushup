import os
import glob
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
from utils import calculate_angle, calculate_hip_deviation, get_active_side, download_pose_model

def extract_features_from_video(landmarker, video_path, label):
    """
    Mengekstrak keypoints dari video, menghitung fitur biomekanika,
    dan mengembalikan list of dict berisi data per frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Gagal membuka video: {video_path}")
        return []
        
    video_data = []
    video_name = os.path.basename(video_path)
    frame_count = 0
    success_count = 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_duration_ms = int(1000 / fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        timestamp_ms = frame_count * frame_duration_ms
        
        # Konversi BGR ke RGB dan bungkus ke mp.Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Deteksi pose
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if results.pose_landmarks:
            success_count += 1
            landmarks = results.pose_landmarks[0]
            side = get_active_side(landmarks)
            
            # Tentukan indeks landmark berdasarkan sisi aktif
            if side == 'left':
                idx_sh, idx_el, idx_wr = 11, 13, 15
                idx_hp, idx_kn, idx_ak = 23, 25, 27
            else:
                idx_sh, idx_el, idx_wr = 12, 14, 16
                idx_hp, idx_kn, idx_ak = 24, 26, 28
                
            sh = [landmarks[idx_sh].x, landmarks[idx_sh].y]
            el = [landmarks[idx_el].x, landmarks[idx_el].y]
            wr = [landmarks[idx_wr].x, landmarks[idx_wr].y]
            hp = [landmarks[idx_hp].x, landmarks[idx_hp].y]
            kn = [landmarks[idx_kn].x, landmarks[idx_kn].y]
            ak = [landmarks[idx_ak].x, landmarks[idx_ak].y]
            
            # Hitung sudut & deviasi
            elbow_angle = calculate_angle(sh, el, wr)
            hip_angle = calculate_angle(sh, hp, kn)
            knee_angle = calculate_angle(hp, kn, ak)
            hip_deviation = calculate_hip_deviation(sh, ak, hp)
            
            # Klasifikasi error_type secara heuristik untuk data label 0
            if label == 1:
                error_type = 'benar'
            else:
                if hip_deviation > 0.04:
                    error_type = 'sagging'
                elif hip_deviation < -0.04:
                    error_type = 'hike'
                else:
                    error_type = 'salah'
                    
            video_data.append({
                'video_name': video_name,
                'frame_idx': frame_count,
                'active_side': side,
                'elbow_angle': round(elbow_angle, 2),
                'hip_angle': round(hip_angle, 2),
                'knee_angle': round(knee_angle, 2),
                'hip_deviation': round(hip_deviation, 4),
                'label': label,
                'error_type': error_type
            })
            
    cap.release()
    return video_data

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    kaggle_dir = os.path.join(dataset_dir, 'kaggle_pushup')
    models_dir = os.path.join(base_dir, 'models')
    pose_model_path = os.path.join(models_dir, 'pose_landmarker_full.task')
    
    categories = {
        'Correct': {
            'path': os.path.join(kaggle_dir, 'Correct sequence'),
            'label': 1
        },
        'Wrong': {
            'path': os.path.join(kaggle_dir, 'Wrong sequence'),
            'label': 0
        }
    }
    
    # Kumpulkan video
    all_videos = []
    for cat_name, info in categories.items():
        if not os.path.exists(info['path']):
            print(f"Error: Folder {info['path']} tidak ditemukan!")
            return
            
        videos = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            videos.extend(glob.glob(os.path.join(info['path'], ext)))
            videos.extend(glob.glob(os.path.join(info['path'], ext.upper())))
            
        print(f"Kategori '{cat_name}': Menemukan {len(videos)} video.")
        
        # Hapus duplikasi path karena case-insensitivity Windows
        videos = list(set(os.path.normpath(v) for v in videos))
        print(f"Kategori '{cat_name}' setelah deduplikasi: {len(videos)} video unik.")
        
        for v in videos:
            all_videos.append((v, info['label']))
            
    if not all_videos:
        print("Tidak ada video yang ditemukan untuk diekstrak fiturnya.")
        return
        
    # Pastikan model pose landmarker tersedia
    download_pose_model(pose_model_path)
    
    # Inisialisasi MediaPipe Pose Landmarker
    print("\nMenginisialisasi MediaPipe Pose Landmarker...")
    base_options = python.BaseOptions(model_asset_path=pose_model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False
    )
    
    all_extracted_data = []
    total_videos = len(all_videos)
    
    for idx, (video_path, label) in enumerate(all_videos):
        video_name = os.path.basename(video_path)
        print(f"[{idx+1}/{total_videos}] Mengekstrak: {video_name} (Label: {label})...")
        
        try:
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                video_data = extract_features_from_video(landmarker, video_path, label)
                all_extracted_data.extend(video_data)
        except Exception as e:
            print(f"Gagal memproses {video_name}: {e}")
            
    if all_extracted_data:
        df = pd.DataFrame(all_extracted_data)
        csv_path = os.path.join(dataset_dir, 'dataset.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Berhasil mengekstrak {len(df)} frame dan menyimpan data ke: {csv_path}")
        print("\nDistribusi Kelas Baru (Data Asli):")
        print(df['error_type'].value_counts())
    else:
        print("\nGagal mengekstrak fitur apa pun dari video.")

if __name__ == '__main__':
    main()
