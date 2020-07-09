# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:13:32 2019

@author: Asad
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

def downsample(filters, size, apply_bn=True, name=None):
    """
    Make downsample block
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = keras.Sequential(name=name)
    result.add(
            layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer = initializer, use_bias=False, name=name+'_conv' if name else name)
            )
    if apply_bn:
        result.add(InstanceNormalization())
    result.add(layers.LeakyReLU(name=name+'_lrelu' if name else name))
    return result

def upsample(filters, size, apply_dropout=False, name=None):
    """
    Make downsample block
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = keras.Sequential(name=name)
    result.add(
            layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer = initializer, use_bias=False, name=name+'_convT' if name else name)
            )
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU(name=name+'_relu' if name else name))
    return result


def Generator():
    down_stack = [
            downsample(32, 4, False, name='down_1'), #bs,192, 192, 32
            downsample(64,4, name='down_2'), # bs, 96,96,64
            downsample(128,4, name='down_3'), # bs, 48,48,128
            downsample(256, 4, name='down_4'), #bs, 24,24,256
            downsample(512,4, name='down_5'), #bs, 12,12,512
            downsample(512,4, name='down_6'), #bs, 6,6,512
            downsample(512,4, name='down_7'), #bs, 3,3,512
            ]
    upstack = [
            upsample(512,4, apply_dropout=True, name='up_7'), #bs, 6,6,512
            upsample(512, 4, apply_dropout=True, name='up_6'), #bs, 12,12,512
            upsample(512,4, apply_dropout=True, name='up_5'), #bs, 24, 24, 512
            upsample(256,4, name='up_4'), #bs, 48,48,256
            upsample(128,4, name='up_3'), #bs, 96,96,128
            upsample(64,4, name='up_2'), #bs,192,192,64
            upsample(32,4, name='up_1') #bs,384,384,32
            ]
    last = layers.Conv2DTranspose(1,4, strides=2, padding='same', kernel_initializer = tf.random_normal_initializer(0.,0.02), activation='tanh', name='last_layer')
    concat = layers.Concatenate()
    
    inputs = keras.layers.Input(shape=[None, None, 1], name='input_layer')
    x = inputs
    
    # downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    
    #upsampling and establishing skip connections
    for up, skip in zip(upstack, skips):
        x = up(x)
        x = concat([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator(target=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')
    x = inp
    if target:
        tar = tf.keras.layers.Input(shape=[None, None, 1], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 384, 384, channels*2)
        
    down1 = downsample(32, 4, False, name='down_1')(x) # (bs, 192, 192, 32)
    down2 = downsample(64, 4, False, name='down_2')(down1) # (bs, 192, 192, 32)
    down3 = downsample(128, 4, name='down_3')(down2) # (bs, 96, 96, 64)
    down4 = downsample(256, 4, name='down_4')(down3) # (bs, 64, 64, 128)
    
    zero_pad1 = tf.keras.layers.ZeroPadding2D(name='zpad_1')(down4) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
    
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
    
    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs = inp, outputs=last)
    
    
def discrimination_loss(real, generated):
    loss_obj = keras.losses.BinaryCrossentropy(from_logits=True)
    
    real_loss = tf.reduce_mean(loss_obj(tf.ones_like(real), real))
    
    generator_loss = tf.reduce_mean(loss_obj(tf.zeros_like(generated), generated))
    
    total_loss = real_loss + generator_loss
    return total_loss*0.5

def generator_loss(generated):
    return tf.reduce_mean(keras.losses.binary_crossentropy(tf.ones_like(generated), generated, from_logits=True) )

def pixel_loss(real, fake, LAMBDA=10):
    loss = tf.reduce_mean(tf.abs(real-fake))
    return loss*LAMBDA*0.5

def identity_loss(real_image, same_image, LAMBDA=10):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss