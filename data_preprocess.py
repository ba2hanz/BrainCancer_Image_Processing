# data_preprocessing.py

import tensorflow as tf
from keras.utils import Sequence
import numpy as np
import pandas as pd
import os
import model_def as md


MODEL_INPUT_SPECS = md.MODEL_INPUT_SPECS


class MultiInputDataSequence(Sequence):
    
    def __init__(self, df, base_dir, batch_size, model_names_list, augment=False, shuffle=True, **kwargs):
        # Keras Sequence için super().__init__ çağrısı
        super().__init__(**kwargs)
        
        self.df = df 
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        
        self.target_sizes = [MODEL_INPUT_SPECS[name] for name in model_names_list]
        self.model_names_list = model_names_list
        
        # Sınıf etiketlerinin sayısal karşılığı
        self.labels = pd.get_dummies(self.df['label']).values 
        
        self.on_epoch_end() # İlk karıştırma

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    # --- YENİ TF İŞLEM FONKSİYONU ---
    def _process_image(self, file_path, target_size, augment):
        # 1. Görseli Yükle (Gri Tonlamalı - 1 kanal)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1) 
        
        # 2. Yeniden Boyutlandırma (Resizing)
        img = tf.image.resize(img, target_size)
        
        # 3. Kanal Çoğaltma (1 -> 3) (RGB'ye benzetme)
        # tf.repeat ile tensörü 3. boyutta 3 kez kopyala
        img = tf.repeat(img, 3, axis=-1)
        
        # 4. Data Augmentation (Sadece eğitim sırasında)
        if augment:
            # Buraya TF'in augmentation fonksiyonları eklenir
            if tf.random.uniform(shape=[]) > 0.5:
                img = tf.image.flip_left_right(img)
            # tf.image.random_brightness, tf.image.random_saturation vb. eklenebilir.
        
        # 5. Normalizasyon (0-1 aralığına)
        img = tf.cast(img, tf.float32) / 255.0
        
        return img.numpy() # NumPy dizisine dönüştürerek döndür

    def __getitem__(self, index):
        indexes = self.df.index[index * self.batch_size : (index + 1) * self.batch_size]
        X_batch_paths = self.df.loc[indexes, 'path'].tolist()
        Y_batch = self.labels[indexes]
        
        # Her bir model girişi için listeler
        multi_input_batch = [[] for _ in self.target_sizes]
        
        for path in X_batch_paths:
            full_path = os.path.join(self.base_dir, path)
            
            for i, target_size in enumerate(self.target_sizes):
                # Görüntüyü o modele özel boyuta işleme
                processed_img = self._process_image(full_path, target_size, self.augment)
                multi_input_batch[i].append(processed_img)
        
        # NumPy array'lerine dönüştür (Batch boyutu, Yükseklik, Genişlik, 3)
        final_inputs = [np.array(input_list) for input_list in multi_input_batch]
        
        # Tek input varsa liste yerine tek numpy array döndür
        # Çoklu input varsa tuple olarak döndür
        if len(final_inputs) == 1:
            return final_inputs[0], np.array(Y_batch)
        else:
            return tuple(final_inputs), np.array(Y_batch)