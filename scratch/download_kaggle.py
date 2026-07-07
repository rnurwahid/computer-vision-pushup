import os
import sys
from dotenv import load_dotenv

# Dynamic path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'kaggle_pushup')

def main():
    print("Loading credentials from .env file...")
    load_dotenv(ENV_PATH)
    
    # Verify environment variables
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    
    if not username or not key:
        print("[ERROR] KAGGLE_USERNAME atau KAGGLE_KEY tidak ditemukan di .env!")
        sys.exit(1)
        
    # Kaggle API reads these environment variables
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    
    try:
        print("Menginisialisasi Kaggle API...")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Unduh dan ekstrak dataset
        dataset_slug = "mohamadashrafsalama/pushup"
        print(f"Mengunduh dataset '{dataset_slug}' ke '{DATASET_DIR}'...")
        os.makedirs(DATASET_DIR, exist_ok=True)
        
        # Download files and unzip
        api.dataset_download_files(dataset_slug, path=DATASET_DIR, unzip=True)
        print("[SUCCESS] Pengunduhan dan ekstraksi dataset Kaggle selesai!")
        sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat mengunduh dari Kaggle: {e}")
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    main()
