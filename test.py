# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:52:39 2019

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

BUFFER_SIZE = 1000
BATCH_SIZE=1
IMG_WIDTH = 512
IMG_HEIGHT = 512

folders = os.listdir(os.path.join('DATASET','IMAGE','TRAINING'))
images, labels = [], []

for f in folders:
    for fname in os.listdir(os.path.join('DATASET','IMAGE','TRAINING',f)):
        images.append(str(os.path.join('DATASET','IMAGE','TRAINING',f,fname)))
        labels.append(str(os.path.join('DATASET','LABEL','TRAINING',f,os.path.splitext(fname)[0]+'.png')))

training_generator = DataGeneratorFull(images, labels, batch_size=BATCH_SIZE, shuffle=False)

#model_id = 'P2P-L1-20190910-143735'
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
ckpt.restore(os.path.join('CKPT',model_id,'ckpt-5'))
os.makedirs(os.path.join('IMAGES','TEST',model_id), exist_ok=True)

def augment_image(img, lbl):
    img, lbl=np.squeeze(img), np.squeeze(lbl)
    from scipy import ndimage
    if np.random.rand()>0.5: #random flip left-right
        img, lbl = np.fliplr(img), np.fliplr(lbl)
    img = ndimage.gaussian_filter(img, sigma=2)
    rotation = np.random.randint(0,15)
    img = ndimage.rotate(img, rotation, reshape=False)
    lbl = ndimage.rotate(lbl, rotation, reshape=False)
    return np.expand_dims(np.expand_dims(img,0),3), np.expand_dims(np.expand_dims(lbl,0),3)

def generate_images(generator, discriminator, img, lbl, n=10):
    plt.figure()
    Image.fromarray(np.uint8(np.squeeze(lbl)*255)).save(os.path.join('IMAGES','lbl.png'))
    Image.fromarray(np.uint8(np.squeeze(img)*255)).save(os.path.join('IMAGES','img.png'))
    for i in range(n):
        us, label = augment_image(img, lbl)
        #prediction = generator([label, rand])
        #d_value = tf.reduce_mean(discriminator([prediction, label]))
        #Image.fromarray(np.uint8(np.squeeze(np.array(prediction))*255)).save(os.path.join('IMAGES','prd-'+str(i+1)+'.png'))
        Image.fromarray(np.uint8(np.squeeze(label)*255)).save(os.path.join('IMAGES','lbl-'+str(i+1)+'.png'))
        Image.fromarray(np.uint8(np.squeeze(us)*255)).save(os.path.join('IMAGES','img-'+str(i+1)+'.png'))
        plt.subplot(1, n, i+1)
        #plt.title(str(np.round(d_value,2)))
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(np.squeeze(np.array(us)))
        plt.axis('off')
    plt.show()
n=1
us, label = training_generator[np.random.randint(0,len(training_generator))]

generate_images(generator, discriminator, us, label)

for us, label in training_generator:
    #generate_images(generator, discriminator, us, label, n=2)
    
    Image.fromarray(np.uint8(np.squeeze(label)*255)).save(os.path.join('IMAGES','TEST',model_id,str(n)+'-lbl.png'))
    Image.fromarray(np.uint8(np.squeeze(us)*255)).save(os.path.join('IMAGES','TEST',model_id,str(n)+'-img.png'))
    rand = tf.random.uniform(shape=label.shape, dtype=label.dtype)
    prediction = generator(label)
    d_value = tf.reduce_mean(discriminator([prediction, label]))
    Image.fromarray(np.uint8(np.squeeze(np.array(prediction))*255)).save(os.path.join('IMAGES','TEST',model_id,str(n)+'-prd.png'))
#    plt.figure()
#    plt.title(str(np.round(d_value,2)))
#    # getting the pixel values between [0, 1] to plot it.
#    plt.imshow(np.squeeze(np.array(prediction)) * 0.5 + 0.5)
#    plt.axis('off')
#    plt.show()
    print('.', end='')
    n+=1
    
    









