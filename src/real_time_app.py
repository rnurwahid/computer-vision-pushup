import os
import sys
import time
import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib

from utils import calculate_angle, calculate_hip_deviation, get_active_side, download_pose_model

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Push-Up Form Analyzer")
    parser.add_argument("--input", type=str, default="0", 
                        help="Path ke file video atau indeks webcam (default: 0 untuk webcam)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path untuk menyimpan video hasil analisis (contoh: output.mp4)")
    parser.add_argument("--depth-threshold", type=float, default=100.0,
                        help="Threshold sudut siku maksimum untuk dianggap push-up penuh/dalam (default: 100)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Path proyek
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'pushup_classifier.joblib')
    pose_model_path = os.path.join(base_dir, 'models', 'pose_landmarker_full.task')
    
    # 1. Muat model ML
    if not os.path.exists(model_path):
        print(f"Error: Model terlatih tidak ditemukan di {model_path}")
        print("Silakan jalankan 'python src/train_model.py' terlebih dahulu untuk membuat model.")
        return
        
    print(f"Memuat model dari: {model_path}...")
    model_data = joblib.load(model_path)
    clf = model_data['model']
    model_name = model_data['model_name']
    feature_cols = model_data['features']
    print(f"Model {model_name} berhasil dimuat (Akurasi latih: {model_data['accuracy']*100:.2f}%)")
    
    # 2. Pastikan file model Pose Landmarker tersedia
    download_pose_model(pose_model_path)
    
    # 3. Inisialisasi input video/webcam
    input_source = args.input
    if input_source.isdigit():
        input_source = int(input_source)
        print(f"Membuka webcam (indeks: {input_source})...")
    else:
        print(f"Membuka file video: {input_source}...")
        
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Gagal membuka input source: {args.input}")
        return
        
    # Ambil spesifikasi video untuk writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30
        
    # Inisialisasi video writer jika path output ditentukan
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))
        print(f"Video hasil analisis akan disimpan ke: {args.output}")
        
    # 4. Inisialisasi MediaPipe Pose Landmarker Tasks API
    print("Menginisialisasi MediaPipe Pose Landmarker...")
    base_options = python.BaseOptions(model_asset_path=pose_model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False
    )
    
    landmarker = vision.PoseLandmarker.create_from_options(options)
    
    # 5. Inisialisasi State Machine & Stats
    # State untuk repetisi: 'UP' (posisi atas/lurus), 'DOWN' (sedang turun/bawah)
    rep_state = 'UP'
    rep_count = 0
    correct_rep_count = 0
    incorrect_rep_count = 0
    
    # Menyimpan prediksi frame dalam repetisi yang sedang berjalan
    current_rep_predictions = []
    current_rep_min_elbow = 180.0
    
    # Feedback detail untuk rep terakhir
    last_rep_feedback = "Lakukan repetisi pertama Anda."
    last_rep_status = "READY"
    
    # Perhitungan FPS harian
    prev_time = 0
    fps_list = []
    frame_index = 0
    
    # Threshold sudut siku untuk mendeteksi repetisi
    ELBOW_THRESHOLD_DOWN = 115.0 # Turun ke bawah threshold ini -> Mulai repetisi
    ELBOW_THRESHOLD_UP = 135.0   # Naik di atas threshold ini -> Repetisi selesai
    
    print("\nMemulai analisis push-up. Tekan 'q' di jendela OpenCV untuk keluar.")
    print("=" * 60)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        frame_index += 1
        
        # Mirroring gambar jika menggunakan webcam
        if isinstance(input_source, int):
            frame = cv2.flip(frame, 1)
            
        h, w, _ = frame.shape
        
        # Konversi BGR ke RGB dan bungkus ke mp.Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Hitung timestamp untuk pendeteksian video
        timestamp_ms = int((frame_index / fps_in) * 1000)
        
        # Deteksi pose
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Default status frame
        frame_posture = "TIDAK TERDETEKSI"
        frame_label = 1
        side = "N/A"
        elbow_ang = 180
        hip_ang = 180
        hip_dev = 0.0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            side = get_active_side(landmarks)
            
            # Tentukan indeks landmark berdasarkan sisi aktif
            if side == 'left':
                idx_sh, idx_el, idx_wr = 11, 13, 15
                idx_hp, idx_kn, idx_ak = 23, 25, 27
            else:
                idx_sh, idx_el, idx_wr = 12, 14, 16
                idx_hp, idx_kn, idx_ak = 24, 26, 28
                
            # Koordinat 2D
            sh = [landmarks[idx_sh].x, landmarks[idx_sh].y]
            el = [landmarks[idx_el].x, landmarks[idx_el].y]
            wr = [landmarks[idx_wr].x, landmarks[idx_wr].y]
            hp = [landmarks[idx_hp].x, landmarks[idx_hp].y]
            kn = [landmarks[idx_kn].x, landmarks[idx_kn].y]
            ak = [landmarks[idx_ak].x, landmarks[idx_ak].y]
            
            # Hitung sudut & deviasi
            elbow_ang = calculate_angle(sh, el, wr)
            hip_ang = calculate_angle(sh, hp, kn)
            knee_ang = calculate_angle(hp, kn, ak)
            hip_dev = calculate_hip_deviation(sh, ak, hp)
            
            # Buat array input untuk prediksi model
            X_frame = np.array([[elbow_ang, hip_ang, knee_ang, hip_dev]])
            frame_label = int(clf.predict(X_frame)[0])
            
            # Klasifikasi jenis postur frame
            if frame_label == 1:
                frame_posture = "POSTUR BENAR (LURUS)"
                color_skeleton = (0, 255, 0) # Hijau
            else:
                # Tentukan jenis kesalahan berdasarkan nilai deviasi
                if hip_dev > 0.04:
                    frame_posture = "PINGGUL MEROSOT (SAGGING)"
                elif hip_dev < -0.04:
                    frame_posture = "PINGGUL NAIK (HIKE)"
                else:
                    frame_posture = "POSTUR SALAH"
                color_skeleton = (0, 0, 255) # Merah
                
            # --- STATE MACHINE REPETISI ---
            if rep_state == 'UP':
                # Jika siku menekuk melewati threshold, kita mulai repetisi baru
                if elbow_ang < ELBOW_THRESHOLD_DOWN:
                    rep_state = 'DOWN'
                    current_rep_predictions = [frame_label]
                    current_rep_min_elbow = elbow_ang
            
            elif rep_state == 'DOWN':
                # Simpan performa posture frame selama di bawah
                current_rep_predictions.append(frame_label)
                if elbow_ang < current_rep_min_elbow:
                    current_rep_min_elbow = elbow_ang
                    
                # Jika siku diluruskan kembali melewati threshold, repetisi selesai
                if elbow_ang > ELBOW_THRESHOLD_UP:
                    rep_state = 'UP'
                    rep_count += 1
                    
                    # 1. Evaluasi kedalaman gerakan (elbow depth)
                    depth_ok = current_rep_min_elbow <= args.depth_threshold
                    
                    # 2. Evaluasi kestabilan postur tubuh (ML prediction)
                    # Jika lebih dari 30% frame salah, maka rep ini dianggap salah postur
                    salah_ratio = current_rep_predictions.count(0) / len(current_rep_predictions)
                    posture_ok = salah_ratio <= 0.30
                    
                    # Tentukan feedback rep terakhir
                    if posture_ok and depth_ok:
                        correct_rep_count += 1
                        last_rep_status = "BENAR"
                        last_rep_feedback = f"Rep {rep_count} BENAR! Postur lurus & kedalaman baik."
                    else:
                        incorrect_rep_count += 1
                        last_rep_status = "SALAH"
                        
                        errors = []
                        if not depth_ok:
                            errors.append("Kurang Dalam (Half-Rep)")
                        if not posture_ok:
                            errors.append("Postur Tubuh Bengkok")
                            
                        last_rep_feedback = f"Rep {rep_count} SALAH: " + ", ".join(errors)
                    
                    print(f"[REPETISI {rep_count}] Status: {last_rep_status} | Min Elbow: {current_rep_min_elbow:.1f} | Feedback: {last_rep_feedback}")

            # Gambar skeleton visual secara custom (lebih bersih)
            # Konversi koordinat normalisasi ke koordinat pixel
            points = {}
            for name, idx in [('sh', idx_sh), ('el', idx_el), ('wr', idx_wr), 
                              ('hp', idx_hp), ('kn', idx_kn), ('ak', idx_ak)]:
                points[name] = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                
            # Gambar titik sendi
            for pt in points.values():
                cv2.circle(frame, pt, 6, (255, 255, 255), -1)
                cv2.circle(frame, pt, 8, color_skeleton, 2)
                
            # Gambar garis skeleton tubuh
            cv2.line(frame, points['sh'], points['el'], color_skeleton, 3)
            cv2.line(frame, points['el'], points['wr'], color_skeleton, 3)
            cv2.line(frame, points['sh'], points['hp'], color_skeleton, 3)
            cv2.line(frame, points['hp'], points['kn'], color_skeleton, 3)
            cv2.line(frame, points['kn'], points['ak'], color_skeleton, 3)

        # 5. Dashboard Visual Overlay (Semi-Transparan)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Hitung FPS real-time
        curr_time = time.time()
        fps = 1.0 / (curr_time - start_time)
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = np.mean(fps_list)
        
        # Tulis Informasi ke Panel Overlay
        cv2.putText(frame, f"REAL-TIME PUSH-UP ANALYZER", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Repetisi: {rep_count} (Benar: {correct_rep_count} | Salah: {incorrect_rep_count})", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        color_text = (0, 255, 0) if frame_label == 1 else (0, 0, 255)
        if results.pose_landmarks is None:
            color_text = (128, 128, 128)
            
        cv2.putText(frame, f"Form Real-time: {frame_posture}", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_text, 2)
        
        cv2.putText(frame, f"Siku: {elbow_ang:.1f} | Pinggul: {hip_ang:.1f} | Deviasi: {hip_dev:.3f}", (20, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        color_rep = (0, 255, 0) if last_rep_status == "BENAR" else (0, 0, 255)
        if last_rep_status == "READY":
            color_rep = (255, 255, 0)
        cv2.putText(frame, f"Status Rep: {last_rep_status}", (20, 155), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rep, 2)
        cv2.putText(frame, f"Feedback: {last_rep_feedback}", (20, 185), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        cv2.putText(frame, f"FPS Perangkat: {avg_fps:.1f} | Sisi Aktif: {side.upper()}", (20, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        if writer:
            writer.write(frame)
            
        try:
            cv2.imshow("Push-up Biomechanical Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error as e:
            if not writer:
                print("GUI tidak terdeteksi. Silakan gunakan argumen --output untuk menyimpan video analisis.")
                break
                
    cap.release()
    if writer:
        writer.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("=" * 60)
    print("ANALISIS SELESAI")
    print(f"Total Repetisi:  {rep_count}")
    print(f"Repetisi Benar:  {correct_rep_count}")
    print(f"Repetisi Salah:  {incorrect_rep_count}")
    if args.output:
        print(f"Video tersimpan di: {args.output}")
    print("=" * 60)

if __name__ == '__main__':
    main()
