import math
import numpy as np
import matplotlib.pyplot as plt
from AE_ML.models.TRFM_AE import TransformerAutoencoder
from AE_ML.util import load_checkpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/dev/null'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import sys

import warnings
warnings.filterwarnings("ignore")

from absl import logging
logging.set_verbosity(logging.ERROR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.callbacks import *

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ..utils.dataset import get_dataloaders
from ..utils.normalization import batch_norm

from ..layers.diffusion import GaussianDiffusion

from tensorflow.keras.models import load_model
tf.config.optimizer.set_jit(False)

class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999):
        super().__init__()
        self.network = network  # denoiser or noise predictor
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema

    def train_step(self, data):
        # Unpack the data
        (images, image_input_past1, image_input_past2), y, improve_past1, improve_past2 = data
        
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]
        
        # 2. Sample timesteps uniformly
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
            print("noise.shape:", noise.shape)
            
            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)
            print("images_t.shape:", images_t.shape)
            
            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t, image_input_past1, image_input_past2, improve_past1, improve_past2], training=True)
            print("pred_noise.shape:", pred_noise.shape)
            
            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

    
    def test_step(self, data):
        # Unpack the data
        (images, image_input_past1, image_input_past2), y, improve_past1, improve_past2 = data

        # 1. Get the batch size
        batch_size = tf.shape(images)[0]
        
        # 2. Sample timesteps uniformly
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        # 3. Sample random noise to be added to the images in the batch
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        
        # 4. Diffuse the images with noise
        images_t = self.gdf_util.q_sample(images, t, noise)
        
        # 5. Pass the diffused images and time steps to the network
        pred_noise = self.network([images_t, t, image_input_past1, image_input_past2, improve_past1, improve_past2], training=False)
        
        # 6. Calculate the loss
        loss = self.loss(noise, pred_noise)

        # 7. Return loss values
        return {"loss": loss}


batch_size = 16
num_epochs = 100
total_timesteps = 120
norm_groups = 8
learning_rate = 1e-4

img_size_H = 96
img_size_W = 144
img_channels = 1

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2 

train_dataset, val_dataset, improve_train, improve_val = get_dataloaders(batch_size=batch_size)

print("Train shape:", train_dataset[0].shape)
print("Val shape:", val_dataset[0].shape)

print("Train improve shape:", improve_train.shape)
print("Val improve shape:", improve_val.shape)

train_data_tf_norm = batch_norm(train_dataset, train_dataset.shape, batch_size=batch_size)
train_data_tf_norm_pred = train_data_tf_norm[2:]
train_data_tf_norm_past1 = train_data_tf_norm[:-2]
train_data_tf_norm_past2 = train_data_tf_norm[1:-1]
improve_past1 = improve_train[:-2]
improve_past2 = improve_train[1:-1]


val_data_tf_norm = batch_norm(val_dataset, val_dataset.shape, batch_size=batch_size)
val_data_tf_norm_pred = val_data_tf_norm[2:]
val_data_tf_norm_past1 = val_data_tf_norm[:-2]
val_data_tf_norm_past2 = val_data_tf_norm[1:-1]
val_improve_past1 = improve_val[:-2]
val_improve_past2 = improve_val[1:-1]

pretrained_encoder = load_model('./CoDiCast/saved_models/aod_encoder.keras')
pretrained_encoder.summary()

# Extract the first 5 layers
first_five_layers = pretrained_encoder.layers[:5]

# Display the first four layers to confirm
for i, layer in enumerate(first_five_layers):
    print(f"Layer {i}: {layer}")

# Create a new model using these layers
# Get the input of the pre-trained model
input_layer = pretrained_encoder.input

# Get the output of the fourth layer
output_layer = first_five_layers[-1].output

# Create the new model
pretrained_encoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Print the summary of the new model
pretrained_encoder.summary()

for layer in pretrained_encoder.layers:
    layer.trainable = False

pretrained_encoder._name = 'encoder'

from ..layers.denoiser import build_unet_model_c2

network = build_unet_model_c2(
    img_size_H=img_size_H,
    img_size_W=img_size_W,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    first_conv_channels=first_conv_channels,
    activation_fn=keras.activations.swish,
    encoder=pretrained_encoder
)

ema_network = build_unet_model_c2(
    img_size_H=img_size_H,
    img_size_W=img_size_W,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    first_conv_channels=first_conv_channels,
    activation_fn=keras.activations.swish,
    encoder=pretrained_encoder
)
ema_network.set_weights(network.get_weights())
print(train_data_tf_norm_pred.shape)
print(train_data_tf_norm_past1.shape)
print(train_data_tf_norm_past2.shape)
print(improve_train.shape)
train_dataset = tf.data.Dataset.from_tensor_slices(((train_data_tf_norm_pred, 
                                                     train_data_tf_norm_past1, 
                                                     train_data_tf_norm_past2,
                                                    ), train_data_tf_norm_pred, improve_past1, improve_past2))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices(((val_data_tf_norm_pred, 
                                                   val_data_tf_norm_past1,
                                                   val_data_tf_norm_past2,
                                                  ), val_data_tf_norm_pred, val_improve_past1, val_improve_past2))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

from ..loss.loss import lat_weighted_loss_mse_56deg

learning_rate = 2e-4
decay_steps = 10000
decay_rate = 0.95


# Get an instance of the Gaussian Diffusion utilities
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# Get the model
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                          decay_steps=decay_steps,
                                                          decay_rate=decay_rate
                                                         )

# Compile the model
model.compile(
              loss=keras.losses.MeanSquaredError(),
              # loss=lat_weighted_loss_mse_56deg,
              optimizer=keras.optimizers.Adam(learning_rate=lr_schedule)
             )

# Train the model
model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=num_epochs,
          batch_size=batch_size
         )

model.save_weights('../checkpoints/aod_model')