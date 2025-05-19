import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import random
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *


from utils.preprocess import batch_norm, batch_norm_inverse
from utils.visuals import vis_one_var_recon
from utils.metrics import lat_weighted_rmse_one_var
from utils.dataset import get_dataloaders

import os

batch_size = 256

train_dataset, val_dataset = get_dataloaders(batch_size=batch_size)

print("Train shape:", train_dataset.shape)
print("Val shape:", val_dataset.shape)

X_train_norm = batch_norm(train_dataset, train_dataset.shape, batch_size=batch_size)
X_val_norm = batch_norm(val_dataset, val_dataset.shape, batch_size=batch_size)

print(X_train_norm.shape, X_val_norm.shape)

def encoder_net(input_shape):
    encoder_inputs = layers.Input(shape=input_shape)

    # Downsampling Path
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same')(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Bottleneck
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    # Upsampling Path (mirroring downsampling)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(1, (3, 3), strides=2, padding='same', activation='relu')(x)

    encoder = keras.Model(encoder_inputs, x, name='encoder_net')
    return encoder

input_shape = (96, 144, 1)
model = encoder_net(input_shape)
model.summary()


learning_rate = 1e-4
decay_steps = 10000
decay_rate = 0.95

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                          decay_steps=decay_steps,
                                                          decay_rate=decay_rate
                                                         )

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse')

# model.compile(optimizer='adam',
#     # optimizer=Adam(learning_rate=lr_schedule),
#              loss='mse', 
#             )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=20)
mc = ModelCheckpoint('../saved_models/aod_encoder.keras', 
                     monitor='val_loss', 
                     mode='min',
                     save_best_only=True,
                    )

model.fit(X_train_norm, X_train_norm, 
          validation_data=(X_val_norm, X_val_norm),
          epochs=100, 
          batch_size=batch_size,
          # verbose=2,
          shuffle=True,
          callbacks=[es, mc]
         )

saved_model = load_model('../saved_models/aod_encoder.keras')

X_recon_norm = saved_model.predict(X_val_norm)
# X_recon_norm.shape
X_recon = batch_norm_inverse(val_dataset, X_recon_norm, X_recon_norm.shape, 1460)

Z500_idx = 0
T850 = 1
T2m_idx = -3
U10_idx = -2
V10_idx = -1

resolution = 2.8125*2

dict = {"Z500":0, "T850":1, "T2m":-3, "U10":-2, "V10":-1}

for var, var_idx in dict.items():
    print(f'{var} RMSE: {lat_weighted_rmse_one_var(X_recon, val_dataset, var_idx=var_idx, resolution=resolution):.2f}')

vis_one_var_recon(X_recon, val_dataset, sample_idx=0, var_idx=0)

lat_weighted_rmse_one_var(X_recon, val_dataset, var_idx=0, resolution=2.8125*2)

vis_one_var_recon(X_recon, val_dataset, sample_idx=0, var_idx=3)

lat_weighted_rmse_one_var(X_recon, val_dataset, var_idx=1, resolution=2.8125*2)

vis_one_var_recon(X_recon, val_dataset, sample_idx=0, var_idx=-2)

lat_weighted_rmse_one_var(X_recon, val_dataset, var_idx=-2, resolution=2.8125*2)

vis_one_var_recon(X_recon, val_dataset, sample_idx=0, var_idx=-1)

lat_weighted_rmse_one_var(X_recon, val_dataset, var_idx=-1, resolution=2.8125*2)

