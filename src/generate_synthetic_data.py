import os
import pandas as pd
import numpy as np

def generate_rep_frames(num_frames, label, error_type):
    """
    Menghasilkan data frame simulasi untuk satu repetisi push-up.
    """
    frames = []
    
    # Simulasikan gerakan naik-turun siku (elbow angle) menggunakan kurva sinus
    # Sudut siku bergerak dari 165 derajat (posisi atas) -> turun -> naik kembali ke 165 derajat
    t = np.linspace(0, np.pi, num_frames)
    
    if error_type == 'benar':
        # Push-up benar: siku turun sampai ~80 derajat (cukup dalam)
        base_elbow = 165 - 85 * np.sin(t)
        # Pinggul lurus: sudut hip di kisaran 172 - 179 derajat
        hip_angles = np.random.uniform(172, 179, num_frames)
        # Lutut lurus: sudut knee di kisaran 172 - 179 derajat
        knee_angles = np.random.uniform(172, 179, num_frames)
        # Deviasi pinggul sangat kecil, mendekati 0
        hip_deviations = np.random.normal(0.0, 0.012, num_frames)
        
    elif error_type == 'sagging':
        # Pinggul melorot ke bawah (sagging)
        base_elbow = 165 - 80 * np.sin(t)
        # Sudut pinggul menekuk ke bawah (135 - 152 derajat)
        hip_angles = np.random.uniform(135, 152, num_frames)
        knee_angles = np.random.uniform(168, 178, num_frames)
        # Deviasi positif (pinggul di bawah garis lurus bahu-ankle)
        hip_deviations = np.random.normal(0.08, 0.015, num_frames)
        
    elif error_type == 'hike':
        # Pinggul menungging ke atas (hip hike)
        base_elbow = 165 - 75 * np.sin(t)
        # Sudut pinggul menekuk ke atas (130 - 150 derajat)
        hip_angles = np.random.uniform(130, 150, num_frames)
        knee_angles = np.random.uniform(165, 175, num_frames)
        # Deviasi negatif (pinggul di atas garis lurus bahu-ankle)
        hip_deviations = np.random.normal(-0.09, 0.015, num_frames)
        
    # Tambahkan noise pada siku agar terlihat natural
    elbow_angles = base_elbow + np.random.normal(0, 2.0, num_frames)
    elbow_angles = np.clip(elbow_angles, 60, 180)
    
    for i in range(num_frames):
        frames.append({
            'video_name': f'simulated_{error_type}_rep.mp4',
            'frame_idx': i + 1,
            'active_side': 'left',
            'elbow_angle': round(float(elbow_angles[i]), 2),
            'hip_angle': round(float(hip_angles[i]), 2),
            'knee_angle': round(float(knee_angles[i]), 2),
            'hip_deviation': round(float(hip_deviations[i]), 4),
            'label': label,
            'error_type': error_type
        })
        
    return frames

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    all_frames = []
    
    # 1. Hasilkan 10 repetisi push-up BENAR (masing-masing 60 frame)
    for _ in range(12):
        all_frames.extend(generate_rep_frames(60, 1, 'benar'))
        
    # 2. Hasilkan 8 repetisi push-up SAGGING (pinggul merosot, masing-masing 60 frame)
    for _ in range(8):
        all_frames.extend(generate_rep_frames(60, 0, 'sagging'))
        
    # 3. Hasilkan 8 repetisi push-up HIKE (pinggul naik, masing-masing 60 frame)
    for _ in range(8):
        all_frames.extend(generate_rep_frames(60, 0, 'hike'))
        
    df = pd.DataFrame(all_frames)
    csv_path = os.path.join(dataset_dir, 'dataset.csv')
    df.to_csv(csv_path, index=False)
    
    print("="*60)
    print("DATASET SIMULASI BERHASIL DICIPTAKAN!")
    print(f"File disimpan di: {csv_path}")
    print(f"Total baris data: {len(df)}")
    print("-"*60)
    print("Distribusi Kelas:")
    print(df['error_type'].value_counts())
    print("="*60)

if __name__ == '__main__':
    main()
