# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:18:24 2019

@author: Asad
"""
import tensorflow as tf
from tensorflow import keras
import pathlib
import functools
from PIL import Image
from PIL import ImageEnhance
from sklearn.utils import shuffle
import numpy as np
from scipy import ndimage
from sklearn.feature_extraction import image

def crop_or_pad(image, t_h, t_w):
    #ipdb.set_trace()
    h, w = image.shape
    if h==t_h and w==t_w:
        return image
    w_d, h_d = t_w-w, t_h-h #calculating height and width difference
    offset_crop_w = np.max([-w_d//2,0])
    offset_pad_w = np.max([w_d//2,0])
    
    offset_crop_h = np.max([-h_d//2,0])
    offset_pad_h = np.max([h_d//2,0])
    
    #maybe needed cropping
    img = image[offset_crop_h:offset_crop_h+t_h, offset_crop_w:offset_crop_w+t_w]
    
    #maybe needed padding
    imgpad = np.zeros((t_h, t_w), image.dtype)
    imgpad[offset_pad_h:offset_pad_h+h, offset_pad_w:offset_pad_w+w] = img
    return imgpad

def prediction_batch(batch_idx, image_files, batch_size, img_height = 512, img_width = 512):
    x_batch = image_files[batch_idx*batch_size:(batch_idx+1)*batch_size] if (batch_idx+1)*batch_size < len(image_files) else image_files[batch_idx*batch_size:]
    x_images = np.expand_dims(np.array([crop_or_pad(np.float32(np.array(Image.open(f).convert('L')))/255.0, img_height, img_width) for f in x_batch]),3)
    return x_images
def prediction_batch_org(batch_idx, image_files, batch_size, img_height = 512, img_width = 512):
    # creates prediction batch without padding or cropping
    x_batch = image_files[batch_idx*batch_size:(batch_idx+1)*batch_size] if (batch_idx+1)*batch_size < len(image_files) else image_files[batch_idx*batch_size:]
    x_images = np.array([np.array(Image.open(f).convert('L')) for f in x_batch])
    return x_images

def augment(x, y):
    if np.random.rand()>0.5: #random flip left-right
        x,y = np.fliplr(x), np.fliplr(y)
    if np.random.rand()>0.5: #random blurring with gaussian noise
        x = ndimage.gaussian_filter(x, sigma=3)
    if np.random.rand()>0.5: #rotate image 0~20 degree randomly
        rotation = np.random.randint(0,15)
        x = ndimage.rotate(x, rotation, reshape=False)
        y = ndimage.rotate(y, rotation, reshape=False)
    return x, y



def create_batch(batch_idx, x_files, y_files, batch_size, is_training=True):
    
    if is_training and batch_idx==0:
        x_files, y_files = shuffle(x_files, y_files)
    
    x_batch = x_files[batch_idx*batch_size:(batch_idx+1)*batch_size] if (batch_idx+1)*batch_size < len(x_files) else x_files[batch_idx*batch_size:]
    y_batch = y_files[batch_idx*batch_size:(batch_idx+1)*batch_size] if (batch_idx+1)*batch_size < len(y_files) else y_files[batch_idx*batch_size:]
    if not is_training:
        x_images = np.expand_dims(np.array([np.float32(np.array(Image.open(f).convert('L')))/255.0 for f in x_batch]),3)
        y_images = np.expand_dims(np.array([np.float32(np.array(Image.open(f).convert('L')))/255.0 for f in y_batch]),3)
    else:
        x_images, y_images = [], []
        for i in range(len(x_batch)):
            x = np.float32(np.array(Image.open(x_batch[i]).convert('L')))/255.0
            y = np.float32(np.array(Image.open(y_batch[i]).convert('L')))/255.0 
            x,y = augment(x,y)
            x_images.append(x)
            y_images.append(y)
        x_images = np.expand_dims(np.array(x_images),3)
        y_images = np.expand_dims(np.array(y_images),3)

    return x_images, y_images


AUTOTUNE = tf.data.experimental.AUTOTUNE

def _process_pathnames(fname, label_path):
    img = tf.image.decode_jpeg(tf.io.read_file(fname), channels=1)
    lbl_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(label_path), channels=1)>127, dtype=tf.uint8)
    return img, lbl_img

def flip_img(img, lbl_img, flip=True):
    if flip:
        flip_prob = tf.random.uniform([], 0.0, 1.0)
        img, lbl_img = tf.cond(tf.less(flip_prob, 0.5),
                               lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(lbl_img)),
                               lambda: (img, lbl_img))
    return img, lbl_img

def _augment(img,
             lbl_img,
             scale=1, # scale image e.g 1/255.
             brightness_delta=0.0, #0.1 #adjust the hue of an 
             contrast_delta = 1.0, #0.9 adjust contrast
             horizontal_flip=False,
             min_jpg_quality=100):
    if brightness_delta:
        img = tf.image.random_brightness(img, brightness_delta)
    if contrast_delta < 1.0:
        img = tf.image.random_contrast(img, contrast_delta, 1.0)
    if min_jpg_quality < 100:
        img = tf.image.random_jpeg_quality(img, min_jpg_quality, 100)
    if horizontal_flip:
        img, lbl_img = flip_img(img, lbl_img)
    #img, lbl_img = _rand_rotate(img, lbl_img)
        
    lbl_img = tf.cast(lbl_img, dtype=tf.float32)
    img = tf.cast(img, dtype=tf.float32) * scale
    return img, lbl_img

def get_baseline_dataset(filenames, 
                         labels,
                         preproc_fn = functools.partial(_augment),
                         threads = AUTOTUNE,
                         batch_size = 4,
                         shuffle=True,
                         repeat=True):
    num_x = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    #process pathnames
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
    
    if shuffle:
        dataset = dataset.shuffle(num_x)
    if repeat: 
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size)
    return dataset
    



def get_dataset(image_path, label_path, batch_size, is_eval_mode=False, threads=AUTOTUNE, repeat=True, shuffle=True):
    images = [str(f) for f in pathlib.Path(image_path).glob('*/*.jpg')]
    labels = [str(f) for f in pathlib.Path(label_path).glob('*/*.jpg')]
    if is_eval_mode:
        val_cfg = {
                'scale': 1/255.
                }
        preprocess_func = functools.partial(_augment, **val_cfg)
        dataset = get_baseline_dataset(images, labels, preproc_fn=preprocess_func, batch_size=batch_size, shuffle=shuffle, threads=threads, repeat=repeat)
    else:
        train_cfg = {
                'scale': 1/255.,
                'horizontal_flip': True,
                'brightness_delta': 0.1,
                'contrast_delta': 0.9,
                'min_jpg_quality': 30,
                }
        preprocess_func = functools.partial(_augment, **train_cfg)
        dataset = get_baseline_dataset(images, labels, preproc_fn=preprocess_func, batch_size=batch_size, shuffle=True, threads=threads, repeat=repeat)
    return dataset, len(images)

#img, lbl = next(iter(dataset))
#plt.figure(figsize=(10,10))
#plt.subplot(1,2,1)
#plt.imshow(img[2,:,:,0])
#plt.subplot(1,2,2)
#plt.imshow(lbl[2,:,:,0])
#plt.show()
#def _parse_training_image(image, label):
#    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(image), channels=1), dtype=tf.float32)
#    label = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(label), channels=1), dtype=tf.float32)
#    # Normalization
#    #image = image/255.0 # normal distribution [-1 +1]
#    #label = label/255.0 
#    
#    # Random flip left-right
#    if np.random.rand() > 0.5:
#        image = tf.image.flip_left_right(image)
#        label = tf.image.flip_left_right(label)
#    
#    #randomly adjust brightness
#    if np.random.rand() > 0.5:
#        image = tf.image.random_brightness(tf.image.random_contrast(tf.image.random_jpeg_quality(image,50, 100), 0.99, 1.0), 0.99, 1.0)
#    
#    return image, label
#
#def _parse_eval_image(image, label):
#    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(image), channels=1), dtype=tf.float32)
#    label = tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.io.read_file(label), channels=1), dtype=tf.float32)
#    # Normalization
##    image = image/255.0 # normal distribution [-1 +1]
##    label = label/255.0 
#    return image, label

#
#train_images = [str(f) for f in pathlib.Path('DATASET/IMAGE/TRAINING').glob('*/*.jpg')]
#train_labels = [str(f) for f in pathlib.Path('DATASET/LABEL/TRAINING').glob('*/*.jpg')]
#
#
#val_images = [str(f) for f in pathlib.Path('DATASET/IMAGE/VALIDATION').glob('*/*.jpg')]
#val_labels = [str(f) for f in pathlib.Path('DATASET/LABEL/VALIDATION').glob('*/*.jpg')]
#
#
#test_images = [str(f) for f in pathlib.Path('DATASET/IMAGE/TESTING').glob('*/*.jpg')]
#test_labels = [str(f) for f in pathlib.Path('DATASET/LABEL/TESTING').glob('*/*.jpg')]
    

class DataGeneratorPatch(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, labels, batch_size=4, dim=(256,256), n_channels=1, shuffle=True, augment=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.images = images
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.images) # int(np.ceil(len(self.images) / self.batch_size))
    
    

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        x_batch = self.images[index] #*self.batch_size:(index+1)*self.batch_size] if (index+1)*self.batch_size < len(self.images) else self.images[index*self.batch_size:]
        y_batch = self.labels[index] #*self.batch_size:(index+1)*self.batch_size] if (index+1)*self.batch_size < len(self.labels) else self.labels[index*self.batch_size:]


        # Generate data
        X, y = self.__data_generation(x_batch, y_batch)

        return X, y
    def _adjust_brightness_contrast(self, img):
        if np.random.rand()>0.5: #random enhancement on scale of 0.5 ~ 1.5
            
            img = ImageEnhance.Contrast(img).enhance(np.random.rand()+0.5)
        if np.random.rand()>0.5:
            
            img = ImageEnhance.Brightness(img).enhance(np.random.rand()+0.5)
        return img
    
    def _augment(self, x, y):
        if np.random.rand()>0.5: #random flip left-right
            
            x,y = np.fliplr(x), np.fliplr(y)
            
        if np.random.rand()>0.5: #random blurring with gaussian noise
            
            x = ndimage.gaussian_filter(x, sigma=3)
            
        if np.random.rand()>0.5: #rotate image 0~15 degree randomly
            
            rotation = np.random.randint(0,15)
            x = ndimage.rotate(x, rotation, reshape=False)
            y = ndimage.rotate(y, rotation, reshape=False)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            self.images, self.labels = shuffle(self.images, self.labels)
            
    def _generate_patch_(self, img, p_h=256, p_w=256, train=True):
        #img: numpy array
        return image.extract_patches_2d(img, (p_h, p_w), max_patches=self.batch_size, random_state=42)
    
    def _generate_blocks(self, img, blocksize=(256,256)):
        M,N = img.shape
        b0, b1 = blocksize
        Mb, Nb = int(np.ceil(M/b0)), int(np.ceil(N/b1))
        if M%b0 > 0:
            tmp = img
            img = np.zeros((Mb*b0, N), dtype=tmp.dtype)
            img[0:M,:] = tmp
        if N%b1 > 0:
            tmp = img
            img = np.zeros((Mb*b0, Nb*b1), dtype=tmp.dtype)
            img[0:M, 0:N] = tmp
        return img.reshape(Mb,b0,Nb,b1).swapaxes(1,2).reshape(-1,b0,b1)
    

    def __data_generation(self, x_batch, y_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if not self.shuffle:
            x_images = np.expand_dims(self._generate_blocks(np.float32(np.array(Image.open(x_batch).convert('L')))/255.0, blocksize=(256,256)),3)
            y_images = np.expand_dims(self._generate_blocks(np.float32(np.array(Image.open(y_batch).convert('L')))/255.0, blocksize=(256,256)),3)
        else:
            x, y = Image.open(x_batch).convert('L'), Image.open(y_batch).convert('L')
            if self.augment:
                x = self._adjust_brightness_contrast(x)
            x = np.float32(np.array(x))/255.0
            y = np.float32(np.array(y))/255.0 
            if self.augment:
                x,y = self._augment(x,y)
            # reshaping to even size
            
                
            x_images = np.expand_dims(np.array(self._generate_patch_(x)),3)
            y_images = np.expand_dims(np.array(self._generate_patch_(y)),3)

        return x_images, y_images


    
class DataGeneratorFull(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, labels, batch_size=4, dim=(512,512), n_channels=1, shuffle=True, augment=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.images = images
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.images) / self.batch_size))
    
    

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        x_batch = self.images[index*self.batch_size:(index+1)*self.batch_size] if (index+1)*self.batch_size < len(self.images) else self.images[index*self.batch_size:]
        y_batch = self.labels[index*self.batch_size:(index+1)*self.batch_size] if (index+1)*self.batch_size < len(self.labels) else self.labels[index*self.batch_size:]


        # Generate data
        X, y = self.__data_generation(x_batch, y_batch)

        return X, y
    def _adjust_brightness_contrast(self, img):
        if np.random.rand()>0.5: #random enhancement on scale of 0.5 ~ 1.5
            
            img = ImageEnhance.Contrast(img).enhance(np.random.rand()+0.5)
        if np.random.rand()>0.5:
            
            img = ImageEnhance.Brightness(img).enhance(np.random.rand()+0.5)
        return img
    
    def _augment(self, x, y):
        if np.random.rand()>0.5: #random flip left-right
            
            x,y = np.fliplr(x), np.fliplr(y)
            
        if np.random.rand()>0.5: #random blurring with gaussian noise
            
            x = ndimage.gaussian_filter(x, sigma=3)
            
        if np.random.rand()>1: #rotate image 0~15 degree randomly
            
            rotation = np.random.randint(0,15)
            x = ndimage.rotate(x, rotation, reshape=False)
            y = ndimage.rotate(y, rotation, reshape=False)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            self.images, self.labels = shuffle(self.images, self.labels)
    

    def __data_generation(self, x_batch, y_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if not self.augment:
            x_images = np.expand_dims(np.array([np.float32(np.array(Image.open(f).convert('L')))/255.0 for f in x_batch]),3)
            y_images = np.expand_dims(np.array([np.float32(np.array(Image.open(f).convert('L')))/255.0 for f in y_batch]),3)
        else:
            x_images, y_images = [], []
            for i in range(len(x_batch)):
                x, y = self._adjust_brightness_contrast(Image.open(x_batch[i]).convert('L')), Image.open(y_batch[i]).convert('L')
                x = np.float32(np.array(x))/255.0
                y = np.float32(np.array(y))/255.0
                x,y = self._augment(x,y)
                x_images.append(x)
                y_images.append(y)
            
            x_images = np.expand_dims(np.array(x_images),3)
            y_images = np.expand_dims(np.array(y_images),3)

        return x_images, y_images
    
class DataGeneratorTest(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, batch_size=1, dim=(512,512), n_channels=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.images = images
        self.n_channels = n_channels

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.images) / self.batch_size))
    
    

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        x_batch = self.images[index*self.batch_size:(index+1)*self.batch_size] if (index+1)*self.batch_size < len(self.images) else self.images[index*self.batch_size:]
        x = np.expand_dims(np.array([np.float32(np.array(Image.open(f).convert('L')))/255.0 for f in x_batch]),3)

        return x, x_batch
    
    
class DataGeneratorCrop(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, labels, batch_size=4, dim=(384,384), n_channels=1, shuffle=True, augment=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.images = images
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.images) / self.batch_size))
    
    

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        x_batch = self.images[index*self.batch_size:(index+1)*self.batch_size] if (index+1)*self.batch_size < len(self.images) else self.images[index*self.batch_size:]
        y_batch = self.labels[index*self.batch_size:(index+1)*self.batch_size] if (index+1)*self.batch_size < len(self.labels) else self.labels[index*self.batch_size:]


        # Generate data
        X, y = self.__data_generation(x_batch, y_batch)

        return X, y
    def _adjust_brightness_contrast(self, img):
        if np.random.rand()>0.5: #random enhancement on scale of 0.5 ~ 1.5
            
            img = ImageEnhance.Contrast(img).enhance(np.random.rand()+0.5)
        if np.random.rand()>0.5:
            
            img = ImageEnhance.Brightness(img).enhance(np.random.rand()+0.5)
        return img
    
    def _augment(self, x, y):
        if np.random.rand()>0.5: #random flip left-right
            
            x,y = np.fliplr(x), np.fliplr(y)
            
        if np.random.rand()>0.5: #random blurring with gaussian noise
            
            x = ndimage.gaussian_filter(x, sigma=3)
            
        if np.random.rand()>0.5: #rotate image 0~15 degree randomly
            
            rotation = np.random.randint(0,15)
            x = ndimage.rotate(x, rotation, reshape=False)
            y = ndimage.rotate(y, rotation, reshape=False)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            self.images, self.labels = shuffle(self.images, self.labels)
    

    def __data_generation(self, x_batch, y_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if not self.shuffle:
            x_images = np.expand_dims(np.array([np.float32(np.array(Image.open(f).convert('L')))/255.0 for f in x_batch]),3)
            y_images = np.expand_dims(np.array([np.float32(np.array(Image.open(f).convert('L')))/255.0 for f in y_batch]),3)
        else:
            x_images, y_images = [], []
            for i in range(len(x_batch)):
                x, y = self._adjust_brightness_contrast(Image.open(x_batch[i]).convert('L')), Image.open(y_batch[i]).convert('L')
                x = np.float32(np.array(x))/255.0
                y = np.float32(np.array(y))/255.0
                x,y = self._augment(x,y)
                x_images.append(x)
                y_images.append(y)
            
            x_images = np.expand_dims(np.array(x_images),3)
            y_images = np.expand_dims(np.array(y_images),3)

        return x_images, y_images