# Push-Up Form Classifier & Biomechanical Analyzer

Sistem kecerdasan buatan berbasis *computer vision* untuk menganalisis, mengklasifikasikan, dan menghitung repetisi gerakan push-up secara *real-time* berbasis sudut sendi biomekanis. Sistem ini menggunakan **MediaPipe Pose Landmarker** untuk pelacakan sendi tubuh dan **Random Forest Classifier** sebagai mesin inferensinya, dioptimalkan secara khusus untuk berjalan sangat ringan pada perangkat CPU saja (uji coba dilakukan pada Lenovo ThinkPad T540p).

Sistem ini melacak parameter biomekanik tubuh per frame untuk mengidentifikasi apakah postur tubuh saat push-up berada dalam posisi **Benar (Lurus)** atau **Salah (seperti Pinggul Merosot / *Sagging* atau Pinggul Naik / *Hike*)**.

---

## Fitur Utama

1.  **Pelacakan Sendi Real-Time:** Menghitung sudut siku, pinggul, lutut, dan penyimpangan pinggul (*hip deviation*) secara *live* menggunakan kamera/webcam atau file video.
2.  **Klasifikasi Akurasi Tinggi (93,78%):** Menggunakan model Random Forest yang dilatih pada dataset gerakan manusia asli untuk mendeteksi gerakan benar vs salah.
3.  **Klasifikasi Kesalahan Spesifik:**
    *   **Pinggul Merosot (*Sagging*):** Akurasi **99,66%**
    *   **Pinggul Naik (*Hike*):** Akurasi **100,00%**
4.  **State Machine Repetisi Pintar:** Menghitung repetisi push-up secara dinamis berbasis pergerakan sudut siku dan mengevaluasi kedalaman push-up (*depth threshold*).
5.  **Optimasi CPU:** Inferensi sangat cepat pada perangkat berdaya rendah tanpa memerlukan GPU eksternal.
6.  **Perekaman Hasil:** Dukungan penyimpanan video hasil analisis lengkap dengan skeleton visual dan dasbor statistik.

---

## Struktur Proyek

```text
ML Project/
├── .gitignore                  # Mengabaikan dataset video besar & kredensial .env
├── README.md                   # File dokumentasi utama ini
├── dataset/
│   ├── README.md               # Dokumentasi sumber dataset UCF101, HMDB51, & Kaggle
│   └── dataset.csv             # Hasil ekstraksi koordinat keypoints biomekanis
├── models/
│   ├── pushup_classifier.joblib # Model Random Forest terpilih (9.6 MB)
│   └── confusion_matrix.png     # Visualisasi grafik performa evaluasi model
└── src/
    ├── real_time_app.py        # Aplikasi dasbor pelacakan push-up real-time
    ├── train_model.py          # Script pelatihan dan pemilihan model terbaik
    └── utils.py                # Fungsi utilitas matematika sudut & unduh model pose
```

---

## Instalasi & Persiapan

### 1. Prasyarat
*   Python 3.10 atau versi di atasnya
*   Terminal PowerShell/Bash

### 2. Kloning Repositori & Instalasi Pustaka
Instal pustaka dependen yang dibutuhkan:
```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib python-dotenv
```

### 3. Konfigurasi Kredensial Kaggle (Opsional - Jika ingin mengunduh ulang)
Jika ingin mengunduh ulang dataset dari Kaggle secara terprogram, buat file bernama `.env` di direktori utama proyek, lalu tambahkan kredensial Kaggle API Anda:
```env
KAGGLE_USERNAME=username_kaggle_anda
KAGGLE_KEY=api_key_kaggle_anda
```

---

## Cara Menjalankan Sistem

Sistem dapat dijalankan langsung menggunakan model terlatih (`pushup_classifier.joblib`) yang sudah disertakan di dalam folder `models/`.

### 1. Menjalankan Aplikasi Pelacakan Push-Up Real-Time
Sistem dapat menggunakan webcam laptop secara langsung atau memasukkan file video:

*   **Menggunakan Webcam (Default):**
    ```bash
    python src/real_time_app.py --input 0
    ```
*   **Menganalisis File Video & Menyimpan Hasilnya:**
    ```bash
    python src/real_time_app.py --input "path/to/your/video.mp4" --output "output/analyzed_video.mp4"
    ```

---

## 🔬 Rekonstruksi & Pelatihan Model dari Awal

Jika ingin mereproduksi eksperimen pelatihan model secara utuh:

### Langkah 1: Unduh Dataset Video
Jalankan script pengunduh untuk mengunduh dataset push-up Kaggle:
```bash
python scratch/download_kaggle.py
```

### Langkah 2: Ekstraksi Fitur Biomekanis
Ekstrak koordinat sendi frame-by-frame menggunakan MediaPipe Pose:
```bash
python src/extract_kaggle_features.py
```
Hasil ekstraksi akan disimpan secara otomatis di file `dataset/dataset.csv`.

### Langkah 3: Latih Model Machine Learning
Latih model Random Forest dan SVM pada dataset yang baru diekstrak:
```bash
python src/train_model.py
```
Script ini akan mencetak laporan klasifikasi (*Classification Report*) dan menyimpan grafik *Confusion Matrix* di `models/confusion_matrix.png`.

---

## Hasil Evaluasi Eksperimen

Pelatihan dilakukan pada **10.925 frame** data berlabel yang seimbang (stratified split 80:20):

| Model | Akurasi |
| :--- | :---: |
| **Random Forest Classifier** | **93,78%** |
| Support Vector Machine (SVM) | 73,27% |

### Akurasi Berdasarkan Jenis Postur:
*   **Lurus (Benar):** **92,01%**
*   **Pinggul Merosot (*Sagging*):** **99,66%**
*   **Pinggul Naik (*Hike*):** **100,00%**
*   **Kesalahan Lainnya:** **87,14%**

Sistem terbukti memiliki ketahanan yang sangat baik dalam membedakan gerakan benar dan salah, serta mampu berjalan dengan performa *framerate* optimal pada spesifikasi CPU menengah ke bawah.

---

## Lisensi
Proyek ini dibuat untuk memenuhi salah satu Tugas Mata Kuliah Machine Learning dan penelitian klasifikasi aktivitas olahraga. 
