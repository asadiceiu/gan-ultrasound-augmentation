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

BUFFER_SIZE = 1000
BATCH_SIZE=1
IMG_WIDTH = 512
IMG_HEIGHT = 512


images = [str(f) for f in pathlib.Path(os.path.join('DATASET','IMAGE', 'TRAINING')).glob('*/*.jpg')]
labels = [str(f) for f in pathlib.Path(os.path.join('DATASET','LABEL', 'TRAINING')).glob('*/*.png')]

training_generator = DataGeneratorFull(images, labels, batch_size=BATCH_SIZE, shuffle=True)

val_images = [str(f) for f in pathlib.Path(os.path.join('DATASET','IMAGE', 'TESTING')).glob('*/*.jpg')]
val_labels = [str(f) for f in pathlib.Path(os.path.join('DATASET','LABEL', 'TESTING')).glob('*/*.png')]

test_generator = DataGeneratorFull(val_images, val_labels, batch_size=BATCH_SIZE, shuffle=False)


gen_label = Generator()
gen_us = Generator()
gen_us.summary()

dis_us = Discriminator(target=True)
dis_label = Discriminator(target=True)

gen_label_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
gen_us_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

dis_us_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
dis_label_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

chekpoint_path = os.path.join('CKPT','TRAIN')

ckpt = tf.train.Checkpoint(
        gen_label               = gen_label,
        gen_us                  = gen_us,
        dis_us                  = dis_us,
        dis_label               = dis_label,
        gen_label_optimizer     = gen_label_optimizer,
        gen_us_optimizer        = gen_us_optimizer,
        dis_us_optimizer        = dis_us_optimizer,
        dis_label_optimizer     = dis_label_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, chekpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest Checkpoint Restored!!')
    
EPOCHS = 50



img_sample, lbl_sample = training_generator[0]

plt.subplot(121)
plt.title('US Image')
plt.imshow(np.squeeze(img_sample))

plt.subplot(122)
plt.title('US Label')
plt.imshow(np.squeeze(lbl_sample))

to_label = gen_label(img_sample)
to_us = gen_us(lbl_sample)

plt.figure(figsize=(8, 8))
contrast = 8

imgs = [img_sample, np.array(to_us), lbl_sample, np.array(to_label)]
title = ['US Image', 'To Label', 'US Label', 'To Image']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(np.squeeze(imgs[i]))
  else:
    plt.imshow(np.squeeze(imgs[i]))
plt.show()


#%% Training Loop
@tf.function
def train_step(real_us, real_label):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator Label translates US -> LABEL
        # Generator US translates LABEL -> US.
        
        # generate fake label from real us
        fake_label = gen_label(real_us, training=True)
        #generate cycled us from generated fake label
        cycled_us = gen_us(fake_label, training=True)
        
        # generate fake us from real label
        fake_us = gen_us(real_label, training=True)
        # generate cycled label from generated fake us
        cycled_label = gen_label(fake_us, training=True)
        
        # discriminate between real label, cycled label and fake label given real us as input
        disc_us_real = dis_us([real_us, real_label], training=True)
        disc_us_fake = dis_us([real_us, fake_label], training=True)
        #disc_us_cycle = dis_us([real_us, cycled_label], training=True)
        
        # discriminate between real us, cycled us and fake us given real label as input
        disc_label_real = dis_label([real_label, real_us], training=True)
        disc_label_fake = dis_label([real_label, fake_us], training=True)
        #disc_label_cycle = dis_label([real_label, cycled_us], training=True)
        
        
        # calculate generator loss for generated label
        gen_label_loss = generator_loss(disc_label_fake)
        # calculate generator loss for recycled label
        #gen_cycle_loss = generator_loss(disc_label_cycle) + generator_loss(disc_us_cycle)
        # calculate generator loss for generated us
        gen_us_loss = generator_loss(disc_us_fake)
        
        
        total_cycle_loss = pixel_loss(real_label, cycled_label) + pixel_loss(real_us, cycled_us)
        
        # Total generator loss = adversarial loss + cycle loss
        total_gen_label_loss = gen_label_loss + total_cycle_loss + pixel_loss(real_label, fake_label)
        total_gen_us_loss = gen_us_loss + total_cycle_loss + pixel_loss(real_us, fake_us)
        
        disc_label_loss = discrimination_loss(disc_label_real, disc_label_fake)
        disc_us_loss = discrimination_loss(disc_us_real, disc_us_fake)
        
        # Calculate the gradients for generator and discriminator
        gen_label_gradients = tape.gradient(total_gen_label_loss, 
                                            gen_label.trainable_variables)
        gen_us_gradients = tape.gradient(total_gen_us_loss, 
                                         gen_us.trainable_variables)
        
        dis_label_gradients = tape.gradient(disc_label_loss, 
                                            dis_label.trainable_variables)
        dis_us_gradients = tape.gradient(disc_us_loss, 
                                         dis_us.trainable_variables)
        # Apply the gradients to the optimizer
        gen_label_optimizer.apply_gradients(zip(gen_label_gradients, 
                                                gen_label.trainable_variables))
        gen_us_optimizer.apply_gradients(zip(gen_us_gradients,
                                             gen_us.trainable_variables))
        
        dis_label_optimizer.apply_gradients(zip(dis_label_gradients,
                                                dis_label.trainable_variables))
        dis_us_optimizer.apply_gradients(zip(dis_us_gradients,
                                             dis_us.trainable_variables))
        
def generate_images(model1, model2, test_input, step):
    prediction = model1(test_input)
    recycled = model2(prediction)
    plt.figure(figsize=(12, 18))
    
    display_list = [np.squeeze(np.array(test_input[0])), np.squeeze(np.array(prediction[0])), np.squeeze(np.array(recycled[0]))]
    title = ['Input Image', 'Predicted Image', 'Recycled Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    Image.fromarray(np.uint8(display_list[0]*255)).save(os.path.join('IMAGES',step+'-input.png'))
    Image.fromarray(np.uint8(display_list[1]*255)).save(os.path.join('IMAGES',step+'-predicted.png'))
    Image.fromarray(np.uint8(display_list[2]*255)).save(os.path.join('IMAGES',step+'-recycled.png'))
  

for epoch in range(EPOCHS,EPOCHS+50):
    start = time.time()
    
    n = 0
    for us, label in training_generator:
        train_step(us, label)
        if n % 10 == 0:
            print('.', end='', flush=True)
        n += 1
    clear_output(wait=True)
    generate_images(gen_label, gen_us, img_sample, str(epoch+1)+'-1')
    generate_images(gen_us, gen_label, lbl_sample, str(epoch+1)+'-2')
    
    if (epoch +1) % 5 ==0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
              ckpt_save_path), flush=True)
    print('Time taken for epoch {} is {} sec\n'.format(epoch+1, time.time()-start), flush=True)