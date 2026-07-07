# Dataset Video untuk Klasifikasi Aksi & Latihan Fisik (Push-up)

Dataset ini berisi kumpulan video aktivitas/aksi manusia dari berbagai sumber internet, yang secara khusus memuat gerakan push-up sebagai bagian dari penelitian klasifikasi bentuk gerakan push-up (correct vs incorrect).

Total video yang diunduh dan disimpan di folder ini adalah **lebih dari 20.100 video (termasuk dataset berlabel dari Kaggle)**.

---

## 1. Daftar Dataset & Link Sumber

### A. HMDB51 (Human Motion Database)
*   **Deskripsi:** Dataset aksi manusia berskala besar dari Brown University, berisi 6.849 klip video aksi yang diambil dari film, video YouTube, dan sumber lainnya. Memiliki 51 kategori aksi, salah satunya adalah **push-up** (`pushup`) dan gerakan olahraga lainnya seperti `situp`, `jump`, `pullup`.
*   **Jumlah Video:** 6.849 video
*   **Format Video:** `.avi`
*   **Sumber Asli:** [Brown University - Serre Lab](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
*   **Link Cermin (Mirror) ZIP:** [Hugging Face - jili5044/hmdb51](https://huggingface.co/datasets/jili5044/hmdb51)
*   **URL File Unduhan:** `https://huggingface.co/datasets/jili5044/hmdb51/resolve/main/hmdb51.zip`
*   **Sitasi Resmi:**
    > Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., & Serre, T. (2011). HMDB: a large video database for human motion recognition. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2556-2563.

### B. UCF101 (Action Recognition Dataset)
*   **Deskripsi:** Dataset aksi manusia dari University of Central Florida, berisi 13.320 klip video dari YouTube dengan 101 kategori aksi yang bervariasi (termasuk olahraga, memainkan instrumen, interaksi manusia, dll.). Salah satu kategorinya adalah **push-up** (`PushUps`) dan latihan fisik lain seperti `SitUps`, `RopeClimbing`, `JumpingJack`.
*   **Jumlah Video:** 13.320 video
*   **Format Video:** `.avi`
*   **Sumber Asli:** [UCF Center for Research in Computer Vision (CRCV)](https://www.crcv.ucf.edu/data/UCF101.php)
*   **Link Cermin (Mirror) ZIP:** [Hugging Face - quchenyuan/UCF101-ZIP](https://huggingface.co/datasets/quchenyuan/UCF101-ZIP)
*   **URL File Unduhan:** `https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip`
*   **Sitasi Resmi:**
    > Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild, CRCV-TR-12-01, November, 2012.

---

## 2. Dataset Tambahan dari Kaggle (Telah Diunduh Secara Lokal)

1.  **LSTM Exercise Classification: Push Up Videos**
    *   **Deskripsi:** Dataset video gerakan push-up yang dilabeli dengan form benar (correct) dan salah (incorrect). Sangat cocok untuk melatih model klasifikasi gerakan push-up.
    *   **Status Lokal:** **Selesai Diunduh** ke folder `dataset/kaggle_pushup/`.
    *   **Jumlah Video:** 100 video (.mp4) (50 Correct Sequence, 50 Wrong Sequence).
    *   **Link Sumber:** [Kaggle - LSTM Exercise Classification: Push Up Videos](https://www.kaggle.com/datasets/mohamadashrafsalama/pushup)

2.  **Roboflow Push-Up Datasets (Referensi Eksternal)**
    *   **Deskripsi:** Komunitas Roboflow menyediakan berbagai dataset deteksi objek dan pose push-up dengan anotasi bounding box.
    *   **Link Sumber:** [Roboflow Universe - Push-up Search](https://universe.roboflow.com/search?q=push-up)

---

## 3. Struktur Folder Dataset Lokal & GitHub

Untuk mencegah kendala unggahan ke GitHub karena batas ukuran file dan jumlah berkas yang terlalu besar (8 GB+), folder dataset diatur dengan ketentuan:
1.  **Diupload ke GitHub:** Hanya folder `dataset/dataset/` (yang berisi berkas CSV ekstraksi fitur) dan `dataset/README.md` ini.
2.  **Diabaikan oleh Git (Git-ignored):** Seluruh berkas video mentah besar (`hmdb51`, `ucf101`, `raw`, dan `kaggle_pushup`). Jika Anda membutuhkan data mentahnya, silakan unduh dari tautan sumber di atas.

Struktur folder lokal:
```text
dataset/
├── README.md               <-- File Dokumentasi Ini (Diupload ke GitHub)
└── dataset.csv             <-- Hasil Ekstraksi Fitur Keypoints (CSV, ~700 KB)
├── raw/                    <-- [Git-ignored] Folder Video Klasifikasi Spesifik
│   ├── benar/
│   ├── salah_hike/
│   └── salah_sagging/
├── hmdb51/                 <-- [Git-ignored] Dataset HMDB51 (6.849 video)
│   ├── pushup/
│   ├── situp/
│   └── ... (51 kategori)
├── ucf101/                 <-- [Git-ignored] Dataset UCF101 (13.320 video)
│   ├── PushUps/
│   ├── SitUps/
│   └── ... (101 kategori)
└── kaggle_pushup/          <-- [Git-ignored] Dataset Gerakan Push-up Berlabel (100 video)
    ├── Correct sequence/
    ├── Wrong sequence/
    └── labels/
```
