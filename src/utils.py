import os
import numpy as np
import requests

def calculate_angle(a, b, c):
    """
    Menghitung sudut (dalam derajat) di vertex b antara titik a, b, dan c.
    a, b, c harus berupa array-like atau list berisi [x, y].
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_hip_deviation(shoulder, ankle, hip):
    """
    Menghitung deviasi vertikal pinggul (hip) dari garis lurus antara bahu (shoulder) dan mata kaki (ankle).
    Semua input berupa list/array [x, y].
    Kembalian:
        - Positif (> 0) jika pinggul merosot ke bawah (sagging hips).
        - Negatif (< 0) jika pinggul menungging ke atas (hip hike).
    """
    x_s, y_s = shoulder[0], shoulder[1]
    x_a, y_a = ankle[0], ankle[1]
    x_h, y_h = hip[0], hip[1]
    
    # Menghindari pembagian dengan nol jika bahu dan mata kaki sejajar vertikal
    if abs(x_a - x_s) < 1e-5:
        return x_h - x_s
    
    # Menghitung y teoritis pada garis bahu-ankle di titik x pinggul
    y_line = y_s + (x_h - x_s) * (y_a - y_s) / (x_a - x_s)
    
    # Deviasi y: karena koordinat gambar y bertambah ke bawah,
    # y_h > y_line berarti pinggul berada di bawah garis (sagging)
    return y_h - y_line

def get_active_side(landmarks):
    """
    Menentukan sisi tubuh mana (kiri atau kanan) yang lebih terlihat di kamera
    berdasarkan rata-rata visibility keypoints dari list landmarks.
    landmarks berupa list dari Landmark objek (panjang 33).
    """
    # Keypoints MediaPipe Pose:
    # Kiri: Shoulder(11), Elbow(13), Wrist(15), Hip(23), Knee(25), Ankle(27)
    # Kanan: Shoulder(12), Elbow(14), Wrist(16), Hip(24), Knee(26), Ankle(28)
    
    left_indices = [11, 13, 15, 23, 25, 27]
    right_indices = [12, 14, 16, 24, 26, 28]
    
    left_visibility = [landmarks[idx].visibility for idx in left_indices]
    right_visibility = [landmarks[idx].visibility for idx in right_indices]
    
    avg_left = np.mean(left_visibility)
    avg_right = np.mean(right_visibility)
    
    return 'left' if avg_left >= avg_right else 'right'

def download_pose_model(dest_path):
    """
    Mengunduh file model MediaPipe Pose Landmarker (.task) jika belum ada secara lokal.
    """
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    if not os.path.exists(dest_path):
        print(f"Model Pose Landmarker tidak ditemukan di: {dest_path}")
        print(f"Mengunduh model dari Google CDN: {url}...")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Pengunduhan model selesai dan berhasil disimpan!")
        except Exception as e:
            print(f"Gagal mengunduh model secara otomatis: {e}")
            print("Silakan unduh secara manual dari URL di atas dan tempatkan di folder 'models/'.")
            raise e
