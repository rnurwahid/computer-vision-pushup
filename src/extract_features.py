import os
import glob
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
from utils import calculate_angle, calculate_hip_deviation, get_active_side, download_pose_model

def extract_features_from_video(landmarker, video_path, label, error_type):
    """
    Membaca video, mengekstrak keypoints menggunakan MediaPipe Tasks API,
    menghitung sudut/deviasi, dan mengembalikan list data frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Gagal membuka video: {video_path}")
        return []
    
    video_data = []
    video_name = os.path.basename(video_path)
    frame_count = 0
    success_count = 0
    
    # Ambil info FPS video untuk menghitung timestamp_ms
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_duration_ms = int(1000 / fps)
    
    print(f"Memproses {video_name} ({error_type})...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        timestamp_ms = frame_count * frame_duration_ms
        
        # Konversi BGR ke RGB dan bungkus ke mp.Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Deteksi pose untuk video (membutuhkan timestamp milidetik)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Di Tasks API, results.pose_landmarks adalah list of lists
        if results.pose_landmarks:
            success_count += 1
            landmarks = results.pose_landmarks[0]
            
            # Deteksi sisi tubuh yang dominan terlihat di kamera
            side = get_active_side(landmarks)
            
            # Indeks keypoints berdasarkan sisi tubuh yang aktif
            if side == 'left':
                idx_sh = 11  # Left Shoulder
                idx_el = 13  # Left Elbow
                idx_wr = 15  # Left Wrist
                idx_hp = 23  # Left Hip
                idx_kn = 25  # Left Knee
                idx_ak = 27  # Left Ankle
            else:
                idx_sh = 12  # Right Shoulder
                idx_el = 14  # Right Elbow
                idx_wr = 16  # Right Wrist
                idx_hp = 24  # Right Hip
                idx_kn = 26  # Right Knee
                idx_ak = 28  # Right Ankle
                
            # Ambil koordinat 2D (x, y)
            sh = [landmarks[idx_sh].x, landmarks[idx_sh].y]
            el = [landmarks[idx_el].x, landmarks[idx_el].y]
            wr = [landmarks[idx_wr].x, landmarks[idx_wr].y]
            hp = [landmarks[idx_hp].x, landmarks[idx_hp].y]
            kn = [landmarks[idx_kn].x, landmarks[idx_kn].y]
            ak = [landmarks[idx_ak].x, landmarks[idx_ak].y]
            
            # Hitung fitur biomekanika
            elbow_angle = calculate_angle(sh, el, wr)
            hip_angle = calculate_angle(sh, hp, kn)
            knee_angle = calculate_angle(hp, kn, ak)
            hip_deviation = calculate_hip_deviation(sh, ak, hp)
            
            # Simpan fitur ke dalam list
            video_data.append({
                'video_name': video_name,
                'frame_idx': frame_count,
                'active_side': side,
                'elbow_angle': round(elbow_angle, 2),
                'hip_angle': round(hip_angle, 2),
                'knee_angle': round(knee_angle, 2),
                'hip_deviation': round(hip_deviation, 4),
                'label': label,           # 1 = Benar, 0 = Salah
                'error_type': error_type  # benar, sagging, hike
            })
            
    cap.release()
    print(f"Selesai: {success_count}/{frame_count} frame berhasil diekstrak.")
    return video_data

def main():
    # Definisikan path folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    raw_dir = os.path.join(dataset_dir, 'raw')
    models_dir = os.path.join(base_dir, 'models')
    pose_model_path = os.path.join(models_dir, 'pose_landmarker_full.task')
    
    categories = {
        'benar': {'path': os.path.join(raw_dir, 'benar'), 'label': 1, 'error_type': 'benar'},
        'salah_sagging': {'path': os.path.join(raw_dir, 'salah_sagging'), 'label': 0, 'error_type': 'sagging'},
        'salah_hike': {'path': os.path.join(raw_dir, 'salah_hike'), 'label': 0, 'error_type': 'hike'}
    }
    
    # Pastikan folder dataset dan sub-foldernya ada
    for cat_name, cat_info in categories.items():
        os.makedirs(cat_info['path'], exist_ok=True)
        
    all_extracted_data = []
    video_found = False
    
    # Ekstraksi fitur dari semua kategori video jika ada video
    for cat_name, cat_info in categories.items():
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(glob.glob(os.path.join(cat_info['path'], ext)))
            video_files.extend(glob.glob(os.path.join(cat_info['path'], ext.upper())))
            
        if len(video_files) > 0:
            video_found = True
            
    if not video_found:
        print("\n" + "="*70)
        print("INFO: Tidak ada file video (.mp4, .avi, dll.) yang ditemukan di folder dataset/raw/.")
        print("Silakan masukkan video push-up Anda ke folder berikut:")
        print(f"  1. Kategori Benar: {categories['benar']['path']}")
        print(f"  2. Kategori Sagging: {categories['salah_sagging']['path']}")
        print(f"  3. Kategori Hip Hike: {categories['salah_hike']['path']}")
        print("\nSebagai alternatif untuk uji coba cepat tanpa video rekam, Anda dapat menjalankan:")
        print("  python src/generate_synthetic_data.py")
        print("untuk membuat dataset simulasi (dataset.csv) agar model bisa segera dilatih.")
        print("="*70 + "\n")
        return

    # Unduh model pose landmarker jika belum ada
    download_pose_model(pose_model_path)
    
    # Inisialisasi Pose Landmarker menggunakan Tasks API
    print("Menginisialisasi MediaPipe Pose Landmarker...")
    base_options = python.BaseOptions(model_asset_path=pose_model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False
    )
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for cat_name, cat_info in categories.items():
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(glob.glob(os.path.join(cat_info['path'], ext)))
                video_files.extend(glob.glob(os.path.join(cat_info['path'], ext.upper())))
                
            for video_path in video_files:
                video_data = extract_features_from_video(
                    landmarker,
                    video_path, 
                    cat_info['label'], 
                    cat_info['error_type']
                )
                all_extracted_data.extend(video_data)
            
    if all_extracted_data:
        # Konversi ke pandas DataFrame
        df = pd.DataFrame(all_extracted_data)
        csv_path = os.path.join(dataset_dir, 'dataset.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nBerhasil menyimpan {len(df)} baris data ke: {csv_path}")
        
        # Tampilkan distribusi kelas
        print("\nDistribusi Data:")
        print(df['error_type'].value_counts())
    else:
        print("Tidak ada pose yang berhasil diekstraksi dari video-video yang ditemukan.")

if __name__ == '__main__':
    main()
