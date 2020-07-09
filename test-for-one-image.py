# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:21:19 2020

@author: Asad
"""

import tensorflow as tf
import numpy as np, os, time
from data_helper import DataGeneratorFull
from cyclegan_model import Generator, Discriminator, discrimination_loss, generator_loss, pixel_loss
import matplotlib.pyplot as plt
from IPython.display import clear_output
from PIL import Image
from datetime import datetime
model_id = 'P2P-L1-20191028-135425' #'P2P-NoL1-20190911-2'

generator = Generator()

discriminator = Discriminator(target=True)

generator_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
chekpoint_path = os.path.join('CKPT',model_id)

ckpt = tf.train.Checkpoint(
        generator                  = generator,
        discriminator              = discriminator,
        generator_optimizer        = generator_optimizer,
        discriminator_optimizer    = discriminator_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, chekpoint_path, max_to_keep=5)
ckpt.restore(os.path.join('CKPT',model_id,'ckpt-6'))
# load label image
label = np.float32(np.array(Image.open(r'D:\Python\US-UNET\CycleGANUS\DATASET\LABEL\TRAINING\US_HU_01\00000005.png').convert('L')))/255.0
# make a batch from label
label = np.expand_dims(np.expand_dims(label,0),3)
prediction = generator(label)
prediction = np.squeeze(prediction)
%varexp --imshow prediction