import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_preprocess import MultiInputDataSequence  # Sequenceimport edildi
from model_def import create_transfer_model, create_hybrid_model, create_custom_model
from model_def import MODEL_INPUT_SPECS # Hibrit model isimlendirmesi için
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import cycle

# ========== GPU YAPILANDIRMASI ==========
print("=" * 60)
print("GPU YAPILANDIRMASI - DETAYLI KONTROL")
print("=" * 60)

# TensorFlow versiyonunu göster
print(f"TensorFlow Versiyonu: {tf.__version__}")
print(f"Keras Versiyonu: {tf.keras.__version__}")

# Tüm fiziksel cihazları listele
print("\nTüm Fiziksel Cihazlar:")
all_devices = tf.config.list_physical_devices()
for device in all_devices:
    print(f"  - {device}")

# GPU'ları listele
gpus = tf.config.list_physical_devices('GPU')
print(f"\nTespit edilen GPU sayısı: {len(gpus)}")

# CUDA ve GPU bilgilerini kontrol et
print("\nCUDA ve GPU Bilgileri:")
import subprocess

# nvidia-smi kontrolü
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ nvidia-smi çalışıyor - GPU driver'lar aktif")
        # GPU bilgilerini göster
        lines = result.stdout.split('\n')
        for line in lines[:8]:  # İlk birkaç satırı göster
            if line.strip():
                print(f"  {line}")
    else:
        print("⚠️  nvidia-smi çalışmıyor")
except FileNotFoundError:
    print("⚠️  nvidia-smi bulunamadı - GPU driver'ları kurulu olmayabilir")
except Exception as e:
    print(f"⚠️  nvidia-smi kontrolü hatası: {e}")

# CUDA kütüphanesini doğrudan kontrol et
print("\nCUDA Kütüphane Kontrolü:")
try:
    # CUDA versiyonunu kontrol et
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ CUDA compiler (nvcc) bulundu")
        lines = result.stdout.split('\n')
        for line in lines[:3]:
            if 'release' in line.lower() or 'version' in line.lower():
                print(f"  {line}")
    else:
        print("⚠️  CUDA compiler bulunamadı")
except FileNotFoundError:
    print("⚠️  CUDA compiler bulunamadı (nvcc PATH'te değil)")
except Exception as e:
    print(f"⚠️  CUDA kontrolü hatası: {e}")

# CUDA_PATH kontrolü
print("\nCUDA Ortam Değişkenleri:")
cuda_path = os.environ.get('CUDA_PATH', 'Belirlenmemiş')
cuda_home = os.environ.get('CUDA_HOME', 'Belirlenmemiş')
print(f"  CUDA_PATH: {cuda_path}")
print(f"  CUDA_HOME: {cuda_home}")
if cuda_path != 'Belirlenmemiş' or cuda_home != 'Belirlenmemiş':
    print("  ✅ CUDA ortam değişkenleri ayarlı")
else:
    print("  ⚠️  CUDA ortam değişkenleri ayarlı değil")

# CUDA kütüphanelerini doğrudan kontrol et (DLL/so dosyaları)
print("\nCUDA Kütüphane Dosyaları:")
try:
    import ctypes
    import platform
    
    if platform.system() == 'Windows':
        # Windows'ta CUDA DLL'lerini kontrol et
        cuda_dll_paths = [
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
            os.path.join(os.environ.get('CUDA_PATH', ''), 'bin'),
            os.path.join(os.environ.get('CUDA_HOME', ''), 'bin'),
        ]
        
        cuda_found = False
        for base_path in cuda_dll_paths:
            if os.path.exists(base_path):
                print(f"  ✅ CUDA klasörü bulundu: {base_path}")
                # Versiyon klasörlerini kontrol et
                try:
                    versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                    if versions:
                        print(f"     Versiyonlar: {', '.join(versions)}")
                        cuda_found = True
                except:
                    pass
        
        if not cuda_found:
            print("  ⚠️  CUDA klasörü bulunamadı")
            
    else:
        # Linux'ta CUDA kütüphanelerini kontrol et
        try:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=5)
            if 'cuda' in result.stdout.lower():
                print("  ✅ CUDA kütüphaneleri sistemde bulundu")
            else:
                print("  ⚠️  CUDA kütüphaneleri sistemde bulunamadı")
        except:
            pass
except Exception as e:
    print(f"  CUDA dosya kontrolü hatası: {e}")

# TensorFlow'un GPU kütüphanelerini kontrol et
print("\nTensorFlow GPU Kütüphaneleri:")
try:
    # GPU kullanılabilir mi
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    print(f"  GPU kullanılabilir: {gpu_available}")
    
    # GPU runtime kontrolü
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"  Mantıksal GPU sayısı: {len(logical_gpus)}")
    
    # CUDA kütüphanesi kontrolü (eğer mevcutsa)
    try:
        # TensorFlow'un CUDA ile build edilip edilmediğini kontrol et
        if hasattr(tf.test, 'is_built_with_cuda'):
            cuda_built = tf.test.is_built_with_cuda()
            print(f"  TensorFlow CUDA ile build edilmiş: {cuda_built}")
            if not cuda_built:
                print("  ⚠️  TensorFlow CUDA desteği ile build edilmemiş!")
                print("     Çözüm: pip uninstall tensorflow")
                print("            pip install tensorflow[gpu]")
                print("            veya")
                print("            pip install tensorflow-gpu")
    except Exception as e:
        print(f"  CUDA build kontrolü hatası: {e}")
    
    # TensorFlow'un CUDA kütüphanelerini bulup bulamadığını kontrol et
    if gpu_available:
        try:
            # Basit bir işlem yaparak GPU'yu test et
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                result = tf.reduce_sum(test_tensor)
                result.numpy()  # Sonucu hesapla
            print(f"  ✅ TensorFlow GPU üzerinde işlem yapabiliyor")
        except Exception as e:
            if "Could not find" in str(e) or "not found" in str(e).lower():
                print(f"  ⚠️  TensorFlow CUDA kütüphanelerini bulamıyor: {e}")
                print("     CUDA/cuDNN kütüphaneleri eksik veya yanlış versiyon olabilir")
            else:
                print(f"  ⚠️  GPU test hatası: {e}")
    
except Exception as e:
    print(f"  Kontrol hatası: {e}")

# GPU'yu zorla kullanmak için ortam değişkenleri ayarla
print("\nGPU Ortam Değişkenleri:")
# CUDA_VISIBLE_DEVICES kontrolü
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Tüm GPU\'lar görünür')
print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")

# GPU'yu zorla kullanmak için (eğer TensorFlow GPU'yu görmüyorsa)
if len(gpus) == 0:
    print("\n⚠️  GPU bulunamadı - Manuel kontrol gerekli:")
    print("   Terminal'de şu komutları çalıştırın:")
    print("   python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"")
    print("   python -c \"import tensorflow as tf; print(tf.test.is_built_with_cuda())\"")

if len(gpus) > 0:
    # Her GPU için memory growth ayarla (tüm belleği hemen ayırmaması için)
    print("\nGPU Yapılandırması:")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✅ GPU: {gpu.name} - Memory Growth: AÇIK")
            
            # GPU detaylarını göster
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"     Detaylar: {details}")
            except:
                pass
        except RuntimeError as e:
            print(f"  ⚠️  GPU yapılandırma hatası: {e}")
            # Eğer zaten yapılandırılmışsa, mevcut yapılandırmayı kullan
            print(f"     Mevcut yapılandırma kullanılacak")
    
    # Varsayılan GPU stratejisi
    print("\n✅ GPU kullanılacak - Eğitim GPU ile yapılacak")
else:
    print("\n⚠️  UYARI: TensorFlow GPU bulamadı!")
    print("\n🔧 DETAYLI SORUN GİDERME:")
    print("\n1. TensorFlow GPU versiyonu kontrolü:")
    print("   python -c \"import tensorflow as tf; print(tf.__version__)\"")
    print("   python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"")
    
    print("\n2. TensorFlow GPU versiyonu yüklü değilse:")
    print("   pip uninstall tensorflow tensorflow-cpu")
    print("   pip install tensorflow[gpu]")
    print("   veya (TensorFlow 2.x için)")
    print("   pip install tensorflow")
    print("   (TensorFlow 2.x genellikle GPU desteği ile gelir)")
    
    print("\n3. CUDA/cuDNN versiyon uyumluluğu:")
    print("   TensorFlow 2.13+ için: CUDA 11.8 ve cuDNN 8.6")
    print("   TensorFlow 2.10-2.12 için: CUDA 11.2-11.8 ve cuDNN 8.1-8.6")
    print("   TensorFlow 2.9 için: CUDA 11.2 ve cuDNN 8.1")
    print("   Kontrol: https://www.tensorflow.org/install/source#gpu")
    
    print("\n4. CUDA PATH ayarları (Windows):")
    print("   CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.x")
    print("   CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.x")
    print("   PATH'e ekle: %CUDA_PATH%\\bin;%CUDA_PATH%\\libnvvp")
    
    print("\n5. cuDNN kontrolü:")
    print("   cuDNN kütüphaneleri CUDA\\bin klasöründe olmalı")
    print("   cudnn64_8.dll (Windows) veya libcudnn.so (Linux)")
    
    print("\n6. GPU driver kontrolü:")
    print("   nvidia-smi komutu çalışmalı")
    print("   Driver versiyonu CUDA ile uyumlu olmalı")
    
    print("\n⚠️  Eğitim CPU ile devam edecek (çok yavaş olabilir)")
    print("💡 Öneri: GPU kurulumunu tamamladıktan sonra programı yeniden başlatın")

print("=" * 60)
print()

# GPU memory log seviyesini ayarla (logları görmek için 1 yapıyoruz - GPU sorunlarını görmek için)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=hepsi, 1=info, 2=warning, 3=error

BASE_DIR = 'data/brain_cancer_data/'
BATCH_SIZE = 32
NUM_CLASSES = 3
EPOCHS = 50

os.makedirs('best_weights', exist_ok=True)
os.makedirs('best_model', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

def create_dataframes(base_dir):
    data = []
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        split_path = os.path.join(base_dir, split)
        # Klasör adları etiketlerinizdir: Brain_Glioma, Brain_Menin, Brain_Tumor
        for label in os.listdir(split_path): 
            label_path = os.path.join(split_path, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    if image_file.endswith(('.jpg', '.jpeg', '.png')):
                        # Mutlak yol yerine, Sequence sınıfı için göreceli yol kaydedilir
                        relative_path = os.path.join(split, label, image_file) 
                        data.append({'path': relative_path, 'label': label, 'split': split})
    
    df = pd.DataFrame(data)
    
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'validation'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"Toplam Görüntü: {len(df)}")
    return train_df, val_df, test_df

train_df, val_df, test_df = create_dataframes(BASE_DIR)
CLASS_LABELS = train_df['label'].unique().tolist()
print(f"Sınıf Etiketleri: {CLASS_LABELS}")



early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

def plot_history(history, model_name):
    """ Accuracy/Loss grafiklerini kaydeder """
    # ... (Önceki yanıttaki plot_history fonksiyonunun kodunu buraya ekleyin)
    plt.figure(figsize=(12, 5))
    # ... (Kod)
    plt.savefig(f'results/plots/{model_name}_loss_accuracy.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    """ Confusion Matrix grafiğini kaydeder """
    # ... (Önceki yanıttaki plot_confusion_matrix fonksiyonunun kodunu buraya ekleyin)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.savefig(f'results/plots/{model_name}_confusion_matrix.png')
    plt.close()

def plot_multiclass_roc(y_true, y_pred_probs, model_name, labels):
    """ Çoklu Sınıf ROC Eğrisini kaydeder ve ortalama AUC döndürür """
    # ... (Önceki yanıttaki plot_multiclass_roc fonksiyonunun kodunu buraya ekleyin)
    n_classes = len(labels)
    y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=n_classes) 
    fpr, tpr, roc_auc = dict(), dict(), dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # ... (Grafik çizimi kodu)
    plt.savefig(f'results/plots/{model_name}_roc_curve.png')
    plt.close()
    return np.mean(list(roc_auc.values()))


MODEL_NAME_LIST = [
    ("EfficientNetV2L_Single", create_transfer_model, tf.keras.applications.EfficientNetV2L, MODEL_INPUT_SPECS['EfficientNetV2L']), 
    ("InceptionResNetV2_Single", create_transfer_model, tf.keras.applications.InceptionResNetV2, MODEL_INPUT_SPECS['InceptionResNetV2']), 
    ("ConvNeXtXLarge_Single", create_transfer_model, tf.keras.applications.ConvNeXtXLarge, MODEL_INPUT_SPECS['ConvNeXtXLarge']), 
    ("DenseNet201_Single", create_transfer_model, tf.keras.applications.DenseNet201, MODEL_INPUT_SPECS['DenseNet201']),
    ("Hybrid_1_EffNet_ConvNext", create_hybrid_model, ['EfficientNetV2L', 'ConvNeXtXLarge'], None),
    ("Hybrid_2_Dense_Incept", create_hybrid_model, ['DenseNet201', 'InceptionResNetV2'], None),
    ("Hybrid_3_EffNet_Dense", create_hybrid_model, ['EfficientNetV2L', 'DenseNet201'], None),
    ("Hybrid_3Model", create_hybrid_model, ['EfficientNetV2L', 'ConvNeXtXLarge', 'InceptionResNetV2'], None),
    ("Hybrid_4Model", create_hybrid_model, ['DenseNet201', 'ConvNeXtXLarge', 'InceptionResNetV2', 'EfficientNetV2L'], None),
    ("Custom_Architecture", create_custom_model, None, MODEL_INPUT_SPECS['Custom_Architecture']),
]

all_results = []
best_model_accuracy = 0.0
best_model_name = ""

for model_name, model_func, model_class, size in MODEL_NAME_LIST:
    # Size bilgisini güvenli şekilde al
    if size is not None:
        print(f"\n======== BAŞLANGIÇ: {model_name} (Giriş: {size[0]}x{size[1]}x3) ========")
    else:
        print(f"\n======== BAŞLANGIÇ: {model_name} ========")

    if model_name.endswith("_Single") or model_name == "Custom_Architecture":
        # Tekli/Custom Modeller için tek boyutlu Sequence
        model_names_list = [model_name.split('_')[0]] if model_name.endswith("_Single") else ["Custom_Architecture"] 
    else:
        # Hibrit Modeller için çoklu boyutlu Sequence
        model_names_list = model_class

    # Veri Sequence'larının Oluşturulması
    train_seq = MultiInputDataSequence(train_df, BASE_DIR, BATCH_SIZE, model_names_list, augment=True)
    val_seq = MultiInputDataSequence(val_df, BASE_DIR, BATCH_SIZE, model_names_list, augment=False)
    test_seq = MultiInputDataSequence(test_df, BASE_DIR, BATCH_SIZE, model_names_list, augment=False, shuffle=False)
    
    # Modeli Oluşturma
    if model_name.endswith("_Single"): 
        model = model_func(model_class, (size[0], size[1], 3)) 
    elif model_name == "Custom_Architecture":
        model = model_func((size[0], size[1], 3))
    else: # Hibrit Modeller
        model = model_func(model_names_list, name=model_name) 
    
    # Model hangi device'da çalışacak kontrol et
    print(f"\nModel Device Bilgisi:")
    try:
        # Model'in ilk katmanının device'ını kontrol et
        first_layer = model.layers[0]
        print(f"  Input Layer: {first_layer.name}")
        # GPU kullanımı için bilgi
        if len(gpus) > 0:
            print(f"  ✅ GPU kullanılacak: {gpus[0].name}")
            print(f"  GPU Memory Growth: Aktif")
        else:
            print(f"  ⚠️  CPU kullanılacak (GPU bulunamadı)")
    except Exception as e:
        print(f"  Device kontrolü: {e}")
    
    # Model Kayıt Callback'i
    model_checkpoint = ModelCheckpoint(f'best_weights/{model_name}.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Eğitimi başlat
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS, 
        callbacks=[early_stopping, lr_scheduler, model_checkpoint],
        verbose=1
    )
    
    # --- DEĞERLENDİRME ---
    
    # a. Grafik Oluşturma
    plot_history(history, model_name)
    
    # b. Tahminleri Al
    test_seq.on_epoch_end()
    y_pred_probs = model.predict(test_seq, steps=len(test_seq))
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # y_true'yu Sequence'dan al ve tahmin boyutuyla eşleştir
    y_true_labels = np.argmax(test_seq.labels, axis=1) 
    y_true = y_true_labels[:len(y_pred)]
    
    # c. Confusion Matrix ve ROC
    plot_confusion_matrix(y_true, y_pred, CLASS_LABELS, model_name)
    avg_auc = plot_multiclass_roc(y_true, y_pred_probs, model_name, CLASS_LABELS)

    # d. Metrikler
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # e. Sonuçları Kaydetme
    result = {
        'Model': model_name,
        'Accuracy': report['accuracy'],
        'Sensitivity (Recall)': report['macro avg']['recall'],
        'Precision': report['macro avg']['precision'],
        'F1-Score': report['macro avg']['f1-score'],
        'Kappa': kappa,
        'AUC': avg_auc 
    }
    all_results.append(result)
    
    # En İyi Modeli Güncelleme
    if result['Accuracy'] > best_model_accuracy:
        best_model_accuracy = result['Accuracy']
        best_model_name = model_name
        model.save(f'best_model/{model_name}_final_best_model.keras') # Tüm projenin en iyi modelini kaydet
        print(f"!!! Yeni En İyi Model Kaydedildi: {model_name}")

# --- G. SONUÇLARIN ÖZETLENMESİ ---
final_df = pd.DataFrame(all_results)
final_df = final_df.sort_values(by='Accuracy', ascending=False)
print("\n========== TÜM MODELLERİN KARŞILAŞTIRMA TABLOSU ==========")
print(final_df.to_markdown(index=False))

# Word belgesine yapıştırmak için CSV kaydı
final_df.to_csv('results/model_comparison_results.csv', index=False)
print(f"\nPROJENİN EN İYİ MODELİ: {best_model_name} (Accuracy: {best_model_accuracy:.4f})")
print("\nTüm grafikler ve sonuçlar 'results' klasörüne kaydedilmiştir.")