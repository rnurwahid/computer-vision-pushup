import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def main():
    # Definisikan path folder proyek
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'dataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"Error: File dataset tidak ditemukan di {dataset_path}")
        print("Silakan jalankan 'python src/extract_features.py' atau 'python src/generate_synthetic_data.py' terlebih dahulu.")
        return
        
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset berhasil dimuat. Total sampel: {len(df)}")
    
    # Tentukan fitur (X) dan target (y)
    # Kita menggunakan sudut siku, sudut pinggul, sudut lutut, dan deviasi pinggul sebagai fitur
    feature_cols = ['elbow_angle', 'hip_angle', 'knee_angle', 'hip_deviation']
    X = df[feature_cols]
    y = df['label']
    
    # Melakukan stratified split (80% Train, 20% Test) agar distribusi kelas tetap seimbang
    # Kita menggunakan kolom 'error_type' sebagai strata agar data latih & uji seimbang untuk semua jenis kesalahan
    X_train, X_test, y_train, y_test, _, test_error_type = train_test_split(
        X, y, df['error_type'], test_size=0.2, random_state=42, stratify=df['error_type']
    )
    
    print(f"Data Latih: {X_train.shape[0]} sampel")
    print(f"Data Uji: {X_test.shape[0]} sampel")
    print("-" * 50)
    
    # 1. Inisialisasi dan latih Random Forest
    print("Melatih Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    # 2. Inisialisasi dan latih SVM
    print("Melatih Support Vector Machine (SVM) Classifier...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    
    print("-" * 50)
    print(f"Akurasi Random Forest: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(f"Akurasi SVM:           {svm_acc:.4f} ({svm_acc*100:.2f}%)")
    print("-" * 50)
    
    # Tentukan model terbaik
    if rf_acc >= svm_acc:
        best_model = rf_model
        best_name = "Random Forest"
        best_preds = rf_preds
        best_acc = rf_acc
    else:
        best_model = svm_model
        best_name = "SVM"
        best_preds = svm_preds
        best_acc = svm_acc
        
    print(f"Model terbaik yang dipilih: {best_name} dengan akurasi {best_acc*100:.2f}%")
    print("\nLaporan Klasifikasi Model Terbaik:")
    print(classification_report(y_test, best_preds, target_names=['Salah (0)', 'Benar (1)']))
    
    # Evaluasi detail berdasarkan jenis kesalahan pada data uji
    test_results_df = pd.DataFrame({
        'error_type': test_error_type,
        'true_label': y_test,
        'pred_label': best_preds
    })
    
    print("Akurasi Deteksi Berdasarkan Kategori Postur:")
    for category in test_results_df['error_type'].unique():
        cat_df = test_results_df[test_results_df['error_type'] == category]
        cat_acc = accuracy_score(cat_df['true_label'], cat_df['pred_label'])
        print(f"  - Kategori '{category}': {cat_acc*100:.2f}%")
    
    # Hasilkan dan Simpan Confusion Matrix
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Salah (0)', 'Benar (1)'],
                yticklabels=['Salah (0)', 'Benar (1)'])
    plt.title(f'Confusion Matrix - {best_name}')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    
    plot_path = os.path.join(models_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nConfusion Matrix disimpan di: {plot_path}")
    
    # Simpan Model Terlatih
    model_path = os.path.join(models_dir, 'pushup_classifier.joblib')
    # Simpan metadatanya juga agar bisa memverifikasi nama model saat dimuat
    joblib.dump({
        'model': best_model,
        'model_name': best_name,
        'features': feature_cols,
        'accuracy': best_acc
    }, model_path)
    print(f"Model terlatih berhasil diekspor ke: {model_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
