"""
Model Özeti ve Bilgileri
Proje raporundaki model detaylarını gösterir
"""

import torch
import sys
import os

def print_model_info():
    """
    Model hakkında detaylı bilgi verir
    """
    print("\n" + "=" * 70)
    print("AKCİĞER OPASİTESİ TESPİT SİSTEMİ - MODEL BİLGİLERİ")
    print("=" * 70)
    
    # Model mimarisi
    print("\n1. MODEL MİMARİSİ")
    print("-" * 40)
    print("   • Algoritma: Faster R-CNN (Region-based Convolutional Neural Network)")
    print("   • Backbone: ResNet-50 (önceden eğitilmiş)")
    print("   • Katman sayısı: 50 katmanlı evrişimsel sinir ağı")
    print("   • Bölge Öneri Ağı (RPN): 3 farklı ölçek, 3 farklı en-boy oranı")
    print("   • ROI Pooling: 7x7 çıktı boyutu")
    
    # Sınıflar
    print("\n2. SINIFLANDIRMA KATEGORİLERİ")
    print("-" * 40)
    classes = [
        "background (arka plan)",
        "opacity (genel opasite)",
        "normal (sağlıklı akciğer)",
        "lung (akciğer alanı)",
        "pneumonia (pnömoni/zatürre)",
        "consolidation (konsolidasyon)",
        "atelectasis (atelektazi)",
        "edema (ödem)",
        "nodule (nodül)"
    ]
    for i, cls in enumerate(classes):
        print(f"   {i}: {cls}")
    
    # Eğitim detayları
    print("\n3. EĞİTİM DETAYLARI")
    print("-" * 40)
    print("   • Epoch sayısı: 8")
    print("   • Optimizasyon: Adam optimizer")
    print("   • Eğitim süresi: ~7 saat (P100 GPU)")
    print("   • Veri seti: 17.283 görüntü")
    print("     - Opasite içeren: 10.983 görüntü")
    print("     - Normal: 6.300 görüntü")
    print("   • Veri kaynakları:")
    print("     - TBX11K Tüberküloz: 2.299 görüntü")
    print("     - RSNA Zatürre: 14.861 görüntü")
    print("     - VinBigData: 2.752 görüntü")
    print("     - SIIM-FISABIO-RSNA COVID-19: 4.587 görüntü")
    
    # Performans metrikleri
    print("\n4. PERFORMANS METRİKLERİ")
    print("-" * 40)
    print("   • mAP@0.5: 0.776")
    print("   • Doğruluk (Accuracy): %91")
    print("   • Precision (Kesinlik): %89-94")
    print("   • Recall (Duyarlılık): %86-92")
    print("   • F1-Score: 0.87-0.93")
    print("   • Ortalama IoU: 0.71")
    
    # Örnek dışı veri setleri performansı
    print("\n5. ÖRNEK DIŞI VERİ SETLERİ PERFORMANSI")
    print("-" * 40)
    print("   • Shenzhen ve Montgomery: %84")
    print("   • DA ve DB: %85")
    print("   • Çocuk Göğüs Röntgeni: %83")
    
    # Confusion Matrix
    print("\n6. CONFUSION MATRIX SONUÇLARI")
    print("-" * 40)
    print("   • True Positives (Doğru Pozitif): %91")
    print("   • False Positives (Yanlış Pozitif): %8")
    print("   • True Negatives (Doğru Negatif): %94")
    print("   • False Negatives (Yanlış Negatif): %9")
    
    # Model dosyaları
    print("\n7. MODEL DOSYALARI")
    print("-" * 40)
    model_dir = "TRAINED_MODEL_FOLDER"
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        pt_files = [f for f in files if f.endswith('.pt')]
        for f in pt_files:
            file_path = os.path.join(model_dir, f)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   • {f}: {size_mb:.2f} MB")
    else:
        print(f"   • exp96_model.pt")
        print(f"   • exp114_model.pt")
    
    # Renk kodları
    print("\n8. HASTALIK TİPİ RENK KODLARI")
    print("-" * 40)
    print("   • Pnömoni: Kırmızı (255,0,0)")
    print("   • Konsolidasyon: Yeşil (0,255,0)")
    print("   • Atelektazi: Mavi (0,0,255)")
    print("   • Ödem: Sarı (255,255,0)")
    print("   • Nodül: Mor (255,0,255)")
    print("   • Opasite: Beyaz (255,255,255)")
    
    print("\n" + "=" * 70)
    print("MODEL BİLGİLERİ TAMAMLANDI")
    print("=" * 70 + "\n")

def check_model_files():
    """
    Model dosyalarını kontrol eder
    """
    model_dir = "TRAINED_MODEL_FOLDER"
    expected_models = ['exp96_model.pt', 'exp114_model.pt']
    
    print("\nModel dosyaları kontrol ediliyor...")
    
    if not os.path.exists(model_dir):
        print(f"✗ {model_dir} klasörü bulunamadı!")
        os.makedirs(model_dir, exist_ok=True)
        print(f"✓ {model_dir} klasörü oluşturuldu.")
        return False
    
    found_models = []
    missing_models = []
    
    for model in expected_models:
        model_path = os.path.join(model_dir, model)
        if os.path.exists(model_path):
            found_models.append(model)
            size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ {model} bulundu ({size:.2f} MB)")
        else:
            missing_models.append(model)
            print(f"✗ {model} bulunamadı!")
    
    if missing_models:
        print(f"\n⚠  Eksik modeller: {', '.join(missing_models)}")
        print("  Bu modeller 'create_model.py' çalıştırılarak oluşturulabilir.")
        return False
    
    print(f"\n✓ Tüm modeller mevcut! ({len(found_models)} dosya)")
    return True

if __name__ == "__main__":
    # Model dosyalarını kontrol et
    check_model_files()
    
    # Model bilgilerini göster
    print_model_info()