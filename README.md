# Beyin Kanseri MRI Sınıflandırma (Glioma / Meningioma / Tumor)

Beyin MR görüntülerini 3 sınıfa ayıran (Brain_Glioma, Brain_Menin, Brain_Tumor) derin öğrenme projesi. Transfer öğrenme, hibrit modelleme ve özel (custom) mimari içerir.

## Özellikler
- Tekli transfer öğrenme modelleri: EfficientNetV2L, InceptionResNetV2, ConvNeXtXLarge, DenseNet201
- Hibrit model: Birden fazla backbone’den global özellik birleştirme
- Özel (custom) mimari: Multi-Scale + Residual + Attention + Depthwise Separable (512x512 giriş)
- Eğitim/Değerlendirme: EarlyStopping, ReduceLROnPlateau, en iyi ağırlıkların kaydı
- Çıktılar: Accuracy/Loss grafikleri, Confusion Matrix, ROC-AUC, sınıflandırma raporu

## Proje Yapısı
```
project/
├─ data/brain_cancer_data/        # (repoya dahil değil)
│  ├─ train/                       # klasör adları sınıf etiketidir
│  │  ├─ Brain_Glioma/
│  │  ├─ Brain_Menin/
│  │  └─ Brain_Tumor/
│  ├─ validation/
│  └─ test/
├─ data_preprocess.py             # Keras Sequence (tekli/çoklu giriş destekli)
├─ model_def.py                   # Transfer/hybrid/custom mimariler
├─ main.py                        # Eğitim, değerlendirme ve kayıt
├─ requirements.txt               # Bağımlılıklar
└─ README.md
```

## Kurulum
### 1) Hızlı (pip)
```bash
pip install -r requirements.txt
```
Not: GPU için Windows’ta pip tek başına CUDA/cuDNN kurmaz. GPU istiyorsanız Conda önerilir.

### 2) Önerilen (Conda + GPU)
```bash
# Ortam oluşturma
conda create -n brain-cancer python=3.11 -y
conda activate brain-cancer

# TensorFlow GPU (Conda CUDA/cuDNN’i otomatik yönetir)
conda install -c conda-forge tensorflow-gpu -y

# Diğer paketler (gerekirse)
pip install -r requirements.txt
```

## Veri Yerleşimi
```
./data/brain_cancer_data/
  train/
    Brain_Glioma/
    Brain_Menin/
    Brain_Tumor/
  validation/
    Brain_Glioma/
    Brain_Menin/
    Brain_Tumor/
  test/
    Brain_Glioma/
    Brain_Menin/
    Brain_Tumor/
```
Klasör adları etiket olarak kullanılır. Desteklenen uzantılar: .jpg/.jpeg/.png

## Çalıştırma
```bash
python main.py
```
- En iyi ağırlıklar: `best_weights/`
- En iyi model: `best_model/`
- Grafikler: `results/plots/`
- Karşılaştırma tablosu: `results/model_comparison_results.csv`

## Custom Model (Özet)
- Giriş: 512x512x3
- Bloklar: Multi-Scale (1x1,3x3,5x5,7x7) + Residual + SE/Spatial Attention + Depthwise Separable
- Global Average + Max Pool birleştirme, ardından sınıflandırma katmanları

## İpuçları
- `BASE_DIR` (`main.py`) varsayılan: `data/brain_cancer_data/`
- Büyük dosyalar repoya dahil edilmez (`.gitignore` ile dışlandı)
- GPU kullanımı: Conda tabanlı kurulum tercih edin

## Sorun Giderme
- GPU bulunamadı:
  - Conda ortamında çalıştığınızdan emin olun: `conda activate brain-cancer`
  - `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
  - Sürücü güncel mi? `nvidia-smi`
- Bellek yetersizliği (OOM):
  - `BATCH_SIZE` küçültün (örn. 16/8)
  - Augmentation ve giriş boyutunu gözden geçirin
- Eğitim çok yavaş:
  - GPU üzerinde çalıştığınızdan emin olun
  - Tekli modelle başlayın; hibritler daha ağırdır

## Lisans
Bu proje eğitim/araştırma amaçlıdır. Veri kaynağınızın lisans koşullarına uyunuz.

