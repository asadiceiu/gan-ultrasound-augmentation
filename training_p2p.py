# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:50:29 2019

@author: Asad
"""

import tensorflow as tf
import numpy as np, os, time, pathlib
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

model_id = 'P2P-L1-'+datetime.now().strftime("%Y%m%d-%H%M%S")
model_id = 'P2P-L1-20191028-135425'
#model_id = 'P2P-NonL1-20190910-144505'
#model_id = 'P2P-NoL1-20190911-2' #latest P2P model on workstation with L1 Loss included
start_epoch = 59
folders = os.listdir(os.path.join('DATASET','IMAGE','TRAINING'))
images = []
labels=[]
for f in folders:
    if f[3]=='H':
        for fname in os.listdir(os.path.join('DATASET','IMAGE','TRAINING',f)):
            images.append(str(os.path.join('DATASET','IMAGE','TRAINING',f,fname)))
            labels.append(str(os.path.join('DATASET','LABEL','TRAINING',f,os.path.splitext(fname)[0]+'.png')))

training_generator = DataGeneratorFull(images, labels, batch_size=BATCH_SIZE, shuffle=True, augment=False)


generator = Generator()

discriminator = Discriminator(target=True)

generator_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# create dirs
os.makedirs(os.path.join('IMAGES',model_id), exist_ok=True)
os.makedirs(os.path.join('CKPT',model_id), exist_ok=True)

chekpoint_path = os.path.join('CKPT',model_id)

ckpt = tf.train.Checkpoint(
        generator                  = generator,
        discriminator                  = discriminator,
        generator_optimizer        = generator_optimizer,
        discriminator_optimizer        = discriminator_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, chekpoint_path, max_to_keep=100)

if start_epoch > 0:
    ckpt.restore(os.path.join('CKPT',model_id,'ckpt-'+str(int(start_epoch-18))))
    print('Latest Checkpoint Restored!!: '+os.path.join('CKPT',model_id,'ckpt-'+str(int(start_epoch-18))))
EPOCHS = 100

tbwriter = tf.summary.create_file_writer(os.path.join('CKPT','LOGS',model_id))


#%% Training Loop
@tf.function
def train_step(real_us, real_label):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator Label translates US -> LABEL
        # Generator US translates LABEL -> US.
        #Random Layer
        
        # generate fake us from real label
        fake_us = generator(real_label, training=True)
        
        # discriminate between real label, cycled label and fake label given real us as input
        discriminator_real = discriminator([real_label, real_us], training=True)
        discriminator_fake = discriminator([real_label, fake_us], training=True)
        #discriminator_cycle = discriminator([real_us, cycled_label], training=True)
        
        
        # calculate generator loss for recycled label
        g_loss = generator_loss(discriminator_fake) + pixel_loss(real_us, fake_us)
        
        d_loss = discrimination_loss(discriminator_real, discriminator_fake)
        
        
        generator_gradients = tape.gradient(g_loss, 
                                         generator.trainable_variables)
        
        discriminator_gradients = tape.gradient(d_loss, 
                                         discriminator.trainable_variables)
        # Apply the gradients to the optimizer
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                             generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                             discriminator.trainable_variables))
    return g_loss, d_loss

def generate_images(model, us, label, step):
    
    prediction = model(label)
    Image.fromarray(np.uint8(np.squeeze(label)*255)).save(os.path.join('IMAGES',model_id,step+'-lbl.png'))
    Image.fromarray(np.uint8(np.squeeze(us)*255)).save(os.path.join('IMAGES',model_id,step+'-img.png'))
    Image.fromarray(np.uint8(np.squeeze(np.array(prediction))*255)).save(os.path.join('IMAGES',model_id,step+'-prd.png'))


for epoch in range(start_epoch,EPOCHS):
    start = time.time()
    
    n = 0
    g_losses, d_losses = [], []
    for us, label in training_generator:
        g_loss, d_loss = train_step(us, label)
        if n % 500 == 0:
            generate_images(generator, us, label, str(epoch)+'-'+str(n+1))
            print('.', end='', flush=True)
        n += 1
        step = epoch*len(training_generator)+n
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        
        if step%200 ==0:
            with tbwriter.as_default():
                tf.summary.scalar('G_loss', g_loss, step = step)
                tf.summary.scalar('D_loss', d_loss, step = step)
                
                
    clear_output(wait=True)
    #generate_images(generator, lbl_sample, str(epoch+1))
    with tbwriter.as_default():
        tf.summary.scalar('Epoch D Loss', np.mean(d_losses), step = epoch)
        tf.summary.scalar('Epoch G Loss', np.mean(g_losses), step = epoch)
    if (epoch +1) % 1 ==0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
              ckpt_save_path), flush=True)
    print('Time taken for epoch {} is {} sec'.format(epoch+1, time.time()-start), flush=True)