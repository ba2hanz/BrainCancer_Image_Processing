from typing import Any


import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate
from keras.layers import BatchNormalization, Activation, Add, Multiply, Lambda, DepthwiseConv2D
from keras.layers import SeparableConv2D, AveragePooling2D
from keras.optimizers import Adam   
from keras.applications import *

NUM_CLASSES = 3
BASE_MODELS_DICT = {
    'ConvNeXtXLarge': ConvNeXtXLarge,
    'DenseNet201': DenseNet201,
    'EfficientNetV2L': EfficientNetV2L,
    'InceptionResNetV2': InceptionResNetV2,
}

MODEL_INPUT_SPECS={
    'ConvNeXtXLarge': (512,512),
    'DenseNet201': (224,224),
    'EfficientNetV2L': (512,512),
    'InceptionResNetV2': (299,299),
    'Custom_Architecture': (512,512),  # Custom model için optimal boyut
}

def create_transfer_model(base_model_class,input_shape,learning_rate=0.001):
    base_model = base_model_class(weights='imagenet',include_top=False,input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)             
    x = Dense(256, activation='relu')(x) # Ekstra yoğun katman
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input,outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def create_hybrid_model(model_name_list,input_shape,name,learning_rate=0.001):

    input_tensors= {}
    pooled_features = []

    for model_name in model_name_list:
        base_model_class = BASE_MODELS_DICT[model_name]
        size = MODEL_INPUT_SPECS[model_name]
        input_shape = (size[0], size[1], 3)

        input_tensor_name = f"input_{model_name}"
        input_tensors[input_tensor_name] = tf.keras.Input(shape=input_shape, name=input_tensor_name)
        
        if not base_model_class:
            raise ValueError(f"Bilinmeyen model ismi: {model_name}")
        
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensors[input_tensor_name],
            input_shape=input_shape,
        )
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        pooled_features.append(x)

    conFeatures = Concatenate()(pooled_features)
    
    x = Dropout(0.5)(conFeatures)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=list(input_tensors.values()),outputs=output,name=name)
    model.compile(optimizer=Adam(learning_rate=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def squeeze_excitation_block(input_tensor, ratio=16):
    """
    Squeeze-and-Excitation (SE) bloğu: Channel attention mekanizması
    Özellik haritalarının önemli kanallarını vurgular
    """
    channels = input_tensor.shape[-1]
    # Squeeze: Global Average Pooling
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // ratio, activation='relu')(se)
    # Excitation: Channel weights
    se = Dense(channels, activation='sigmoid')(se)
    # Scale: Özellik haritalarını ağırlıklandır
    se = tf.keras.layers.Reshape((1, 1, channels))(se)
    return Multiply()([input_tensor, se])

def spatial_attention_block(input_tensor):
    """
    Spatial Attention bloğu: Önemli uzamsal bölgeleri vurgular
    """
    # Channel-wise average ve max pooling
    avg_out = Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(input_tensor)
    max_out = Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(input_tensor)
    # Birleştir ve convolution uygula
    concat = Concatenate(axis=3)([avg_out, max_out])
    attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
    return Multiply()([input_tensor, attention])

def multi_scale_block(input_tensor, filters):
    """
    Multi-Scale Feature Extraction: Farklı kernel boyutlarıyla paralel convolution
    Beyin tümörlerinin farklı boyutlardaki özelliklerini yakalar
    """
    # 1x1 convolution - pointwise
    branch1x1 = Conv2D(filters // 4, (1, 1), padding='same')(input_tensor)
    branch1x1 = BatchNormalization()(branch1x1)
    branch1x1 = Activation('relu')(branch1x1)
    
    # 3x3 convolution - küçük detaylar için
    branch3x3 = Conv2D(filters // 4, (3, 3), padding='same')(input_tensor)
    branch3x3 = BatchNormalization()(branch3x3)
    branch3x3 = Activation('relu')(branch3x3)
    
    # 5x5 convolution - orta ölçekli özellikler için
    branch5x5 = Conv2D(filters // 4, (5, 5), padding='same')(input_tensor)
    branch5x5 = BatchNormalization()(branch5x5)
    branch5x5 = Activation('relu')(branch5x5)
    
    # 7x7 convolution - büyük ölçekli özellikler için (tümör boyutları için önemli)
    branch7x7 = Conv2D(filters // 4, (7, 7), padding='same')(input_tensor)
    branch7x7 = BatchNormalization()(branch7x7)
    branch7x7 = Activation('relu')(branch7x7)
    
    # Tüm branch'leri birleştir
    output = Concatenate(axis=3)([branch1x1, branch3x3, branch5x5, branch7x7])
    return output

def residual_block(input_tensor, filters, use_se=True, use_spatial_att=True):
    """
    Residual Block: Derin öğrenme için skip connection
    SE ve Spatial Attention ile güçlendirilmiş
    """
    # Ana yol
    x = Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Attention mekanizmaları
    if use_se:
        x = squeeze_excitation_block(x)
    if use_spatial_att:
        x = spatial_attention_block(x)
    
    # Skip connection
    # Giriş ve çıkış boyutları farklıysa projection
    if input_tensor.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def depthwise_separable_block(input_tensor, filters):
    """
    Depthwise Separable Convolution: Verimli ve hafif convolution
    """
    # Depthwise convolution
    x = DepthwiseConv2D((3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Pointwise convolution
    x = Conv2D(filters, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def create_custom_model(input_shape, learning_rate=0.001):
    """
    Özgün Custom Model: Beyin Kanseri Sınıflandırması için Özel Tasarım
    
    Mimari Özellikleri:
    1. Multi-Scale Feature Extraction: Farklı boyutlardaki tümör özelliklerini yakalar
    2. Residual Connections: Derin öğrenme için gradient akışını iyileştirir
    3. Attention Mechanisms: Önemli özellikleri vurgular (Channel + Spatial)
    4. Depthwise Separable Convolutions: Verimli özellik çıkarımı
    5. Progressive Feature Refinement: Katmanlar arası özellik iyileştirme
    
    Bu model beyin MR görüntülerindeki ince detayları ve farklı ölçeklerdeki
    tümör özelliklerini yakalamak için optimize edilmiştir.
    """
    
    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name='input')
    
    # İlk Convolution Bloğu - Temel özellik çıkarımı
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Stage 1: Multi-Scale Feature Extraction + Residual Blocks
    x = multi_scale_block(x, filters=128)
    x = Dropout(0.2)(x)
    x = residual_block(x, filters=128, use_se=True, use_spatial_att=True)
    
    # Stage 2: Depthwise Separable + Residual
    x = depthwise_separable_block(x, filters=256)
    x = MaxPooling2D((2, 2))(x)
    x = residual_block(x, filters=256, use_se=True, use_spatial_att=True)
    x = Dropout(0.3)(x)
    
    # Stage 3: Multi-Scale + Residual (Derin özellikler)
    x = multi_scale_block(x, filters=512)
    x = residual_block(x, filters=512, use_se=True, use_spatial_att=True)
    x = residual_block(x, filters=512, use_se=True, use_spatial_att=False)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)
    
    # Stage 4: Final Feature Refinement
    x = depthwise_separable_block(x, filters=512)
    x = residual_block(x, filters=512, use_se=True, use_spatial_att=True)
    x = AveragePooling2D((2, 2))(x)
    
    # Global Feature Aggregation: Hem average hem max pooling kullan
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    x = Concatenate()([gap, gmp])
    
    # Classification Head
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x)
    
    # Model oluştur
    model = Model(inputs=inputs, outputs=outputs, name='CustomBrainCancerModel')
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model