import os
import sys
import time
import requests
import zipfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Target directories (dynamic configuration)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

URLS = {
    "HMDB51": {
        "url": "https://huggingface.co/datasets/jili5044/hmdb51/resolve/main/hmdb51.zip",
        "zip_name": "hmdb51.zip",
        "extract_dir": os.path.join(DATASET_DIR, 'hmdb51')
    },
    "UCF101": {
        "url": "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip",
        "zip_name": "UCF-101.zip",
        "extract_dir": os.path.join(DATASET_DIR, 'ucf101')
    }
}

def download_file_resumable(url, zip_path, num_connections=20, chunk_size=5*1024*1024):
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. Dapatkan ukuran total file
    response = requests.head(url, allow_redirects=True, headers=headers)
    response.raise_for_status()
    total_size = int(response.headers.get("Content-Length", 0))
    print(f"[DOWNLOAD] Ukuran total file: {total_size / (1024**2):.2f} MB")
    
    # Buat folder sementara untuk part file
    parts_dir = zip_path + "_parts"
    os.makedirs(parts_dir, exist_ok=True)
    
    # 2. Bagi file menjadi beberapa chunk
    ranges = []
    start = 0
    while start < total_size:
        end = min(start + chunk_size - 1, total_size - 1)
        ranges.append((start, end))
        start += chunk_size
        
    total_chunks = len(ranges)
    print(f"[DOWNLOAD] Membagi file menjadi {total_chunks} chunk berukuran {chunk_size / (1024**2):.1f} MB.")
    
    # Temukan chunk mana saja yang sudah lengkap terunduh
    completed_chunks = 0
    chunks_to_download = []
    
    for i, (start_offset, end_offset) in enumerate(ranges):
        part_path = os.path.join(parts_dir, f"part_{i}")
        expected_size = end_offset - start_offset + 1
        
        if os.path.exists(part_path) and os.path.getsize(part_path) == expected_size:
            completed_chunks += 1
        else:
            chunks_to_download.append((i, start_offset, end_offset))
            
    print(f"[DOWNLOAD] Progress sebelumnya: {completed_chunks}/{total_chunks} chunk sudah ada. Tersisa {len(chunks_to_download)} chunk untuk diunduh.")
    
    if not chunks_to_download:
        print("[DOWNLOAD] Semua chunk sudah lengkap di disk!")
        return parts_dir, total_chunks
        
    downloaded_count = [completed_chunks]
    start_time = time.time()
    
    def download_chunk(chunk_idx, start_offset, end_offset):
        part_path = os.path.join(parts_dir, f"part_{chunk_idx}")
        chunk_headers = {"Range": f"bytes={start_offset}-{end_offset}", "User-Agent": "Mozilla/5.0"}
        
        for attempt in range(5):
            try:
                res = requests.get(url, headers=chunk_headers, timeout=25)
                res.raise_for_status()
                with open(part_path, "wb") as f:
                    f.write(res.content)
                return len(res.content)
            except Exception as e:
                print(f"[RETRY] Chunk {chunk_idx} (percobaan {attempt+1}) gagal: {e}")
                time.sleep(2)
        raise Exception(f"Gagal mengunduh Chunk {chunk_idx} setelah 5 percobaan.")
        
    last_reported_percent = [-5]
    
    with ThreadPoolExecutor(max_workers=num_connections) as executor:
        futures = {
            executor.submit(download_chunk, idx, start_off, end_off): idx 
            for idx, start_off, end_off in chunks_to_download
        }
        
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                future.result()
                downloaded_count[0] += 1
                percent = int(downloaded_count[0] * 100 / total_chunks)
                if percent - last_reported_percent[0] >= 5:
                    elapsed = time.time() - start_time
                    downloaded_bytes = (downloaded_count[0] - completed_chunks) * chunk_size
                    speed = downloaded_bytes / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                    print(f"[DOWNLOAD PROGRESS] {percent}% ({downloaded_count[0]}/{total_chunks} chunk), Rata-rata Kecepatan: {speed:.2f} MB/s, Waktu: {elapsed:.1f}s")
                    last_reported_percent[0] = percent
                    sys.stdout.flush()
            except Exception as e:
                print(f"[FATAL] Gagal pada chunk {chunk_idx}: {e}")
                sys.stdout.flush()
                raise e
                
    return parts_dir, total_chunks

def merge_parts(parts_dir, zip_path, total_chunks):
    print(f"\n[MERGE] Menggabungkan {total_chunks} part file menjadi {zip_path}...")
    with open(zip_path, "wb") as outfile:
        for i in range(total_chunks):
            part_path = os.path.join(parts_dir, f"part_{i}")
            with open(part_path, "rb") as infile:
                outfile.write(infile.read())
    print("[MERGE DONE] Penggabungan selesai!")
    
    # Hapus folder part sementara
    print(f"[CLEANUP] Menghapus folder part sementara: {parts_dir}")
    shutil.rmtree(parts_dir)

def extract_zip(zip_path, extract_dir):
    print(f"\n[EXTRACTION] Mengekstrak {zip_path} ke {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        print(f"[EXTRACTION] Menemukan {total_files} file di dalam ZIP.")
        
        extracted_count = 0
        last_reported_percent = -10
        
        for file in file_list:
            zip_ref.extract(file, extract_dir)
            extracted_count += 1
            percent = int(extracted_count * 100 / total_files)
            if percent - last_reported_percent >= 10:
                print(f"[EXTRACTION PROGRESS] {percent}% ({extracted_count}/{total_files} file)")
                last_reported_percent = percent
                sys.stdout.flush()
                
    print("[EXTRACTION DONE] Ekstraksi selesai!")

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    for name, info in URLS.items():
        print("\n" + "="*80)
        print(f"MEMPROSES DATASET: {name}")
        print("="*80)
        sys.stdout.flush()
        
        zip_path = os.path.join(DATASET_DIR, info["zip_name"])
        extract_dir = info["extract_dir"]
        
        # Cek jika folder ekstraksi sudah terisi
        if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
            print(f"[SKIP] Folder {extract_dir} sudah terisi. Melewati proses.")
            sys.stdout.flush()
            continue
            
        try:
            # 1. Download chunks secara resumable
            parts_dir, total_chunks = download_file_resumable(info["url"], zip_path, num_connections=20)
            
            # 2. Gabungkan part menjadi file ZIP utuh
            merge_parts(parts_dir, zip_path, total_chunks)
            
            # 3. Ekstrak file ZIP
            extract_zip(zip_path, extract_dir)
            
            # 4. Bersihkan file ZIP
            print(f"[CLEANUP] Menghapus file ZIP: {zip_path}")
            os.remove(zip_path)
            sys.stdout.flush()
            
        except Exception as e:
            print(f"[ERROR] Gagal memproses {name}: {e}")
            sys.stdout.flush()
            # Di sini kita TIDAK menghapus folder parts_dir agar bisa di-resume nanti!
            continue
            
    print("\n" + "="*80)
    print("PROSES ALUR DATASET SELESAI DENGAN SUKSES!")
    print("="*80)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
