#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image
import cv2
from datetime import datetime
import argparse
import random

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate
from keras import backend as K
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance


data_gen_args = dict(
        rotation_range=10.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
)

def showOpencvImage(image, isGray=False):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap = 'gray')
    plt.show()

def readBinaryData(n,SIZE,H,nbytes):

    if nbytes==2:
        d = np.zeros((SIZE,SIZE,H),np.uint16)
    elif nbytes==1:
        d = np.zeros((SIZE,SIZE,H),np.uint8)
    else:
        print('Wrong number of bytes per voxel')
        return
    
    f=open(n,"rb")
    for i in range(0,H):
        for j in range(0,SIZE):
            for k in range(0,SIZE):
                byte = f.read(nbytes)
                if nbytes==2:
                    a = 256*byte[0] + byte[1]
                else:
                    a = byte[0]
                d[j,k,i] = a
    f.close()
    return d

def get_augmented(
    X_train, 
    Y_train, 
    X_val=None,
    Y_val=None,
    batch_size=32, 
    seed=0, 
    data_gen_args = dict(
        rotation_range=10.,
        width_shift_range=0.02,
        height_shift_range=0.02,
        zca_whitening = False,
        zca_epsilon = 1e-6,
        shear_range=5,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )):


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)
    
    train_generator = zip(X_train_augmented, Y_train_augmented)
    return train_generator



def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

def conv2d_block(
    inputs, 
    filters=16, 
    kernel_size=(3,3), 
    activation='tanh', 
#    kernel_initializer='he_normal', 
    kernel_initializer= 'glorot_uniform',
    padding='same'):
    
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (inputs)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding) (c)
    return c

def my_custom_unet(
    input_shape,
    num_classes=1,
    upsample_mode='deconv', # 'deconv' or 'simple' 
    filters=16,
    num_layers=4,
    output_activation='softmax'): # 'sigmoid' or 'softmax'
    
    if upsample_mode=='deconv':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs   

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        filters = filters*2 # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=filters)


    for conv in reversed(down_layers):        
        filters //= 2 # decreasing number of filters with each layer 
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same') (x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters)
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation) (x)    

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)




config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
SEED = 10

ap = argparse.ArgumentParser()
ap.add_argument("-c","--cv", required=True,
	help="batch id, from 0 to 4 included")
args = vars(ap.parse_args())

LES_DIR = '/home/user/Spine/Data/GroundTruth_MajorityVoting/'

numbers = [name.split('_')[2] for name in glob.glob(LES_DIR + 'MV*.raw')]

random.seed(SEED)
random.shuffle(numbers)

cv = int(args["cv"])
step = len(numbers)//5
st = cv*step
en = st + step

numbers = numbers[0:st] + numbers[en:]

print(numbers)

T1_DIR = '/home/user/Spine/Data/T1/'
LAB_DIR = '/home/user/Spine/Data/BoneLabels/'

t1_dir_list = [n for n in glob.glob(T1_DIR+'T1*.raw') if n.split('_')[1] in numbers]
t1_dir_list.sort()    

lab_dir_list = [n for n in glob.glob(LAB_DIR+'Bone*.raw') if n.split('_')[1] in numbers]
lab_dir_list.sort()

print(t1_dir_list)
print(lab_dir_list)

imgs_list = []
masks_list = []
size = (512,512)

for t,l in zip(t1_dir_list,lab_dir_list):
    
    name_t = t.split('/')[-1].split('.')[0]
    name_l = l.split('/')[-1].split('.')[0]
    _ , num_t,SIZE_t,_,H_t,B_t,_ = name_t.split('_')
    _ , num_l,SIZE_l,_,H_l,B_l,_ = name_l.split('_')
    if num_t != num_l or SIZE_t != SIZE_l or H_t!= H_l or int(B_l)!=1:
        raise ValueError
    
    t1 = readBinaryData(t,int(SIZE_t),int(H_t),int(B_t))
    lab = readBinaryData(l,int(SIZE_l),int(H_l),int(B_l))
    FAC_T1 = 255./ np.max(t1)
    
    for s in range(0,int(H_t)):
        if np.max(lab[:,:,s]) > 0:
            im = cv2.resize(t1[:,:,s]*FAC_T1,size)
            imgs_list.append(im)
            im = cv2.resize(lab[:,:,s],size,interpolation=cv2.INTER_NEAREST)
            imMask = np.zeros((im.shape[0],im.shape[1],4),dtype=np.float32)
            imMask[im==0,0] = 1               #background
            imMask[im==1,1] = 1              #left 
            imMask[im==2,2] = 1             #middle
            imMask[im==3,3] = 1             #right
            masks_list.append(imMask)

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)
print(imgs_np.shape, masks_np.shape)

weights = np.ones((4),dtype=np.float32)
for i in range(0,4):
    weights[i] = 1/(np.sum(masks_np[:,:,:,i])/(masks_np.shape[0]*masks_np.shape[1]*masks_np.shape[2]))

w = sum(weights)
weights = weights/w

print(weights)

print(imgs_np.max(), masks_np.max())
x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)
print(x.max(), y.max())
print(x.shape, y.shape)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 4)
print(x.shape, y.shape)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
print(x.shape, y.shape)

VAL_SPLIT = 0.8

NUM_TRAIN = int(VAL_SPLIT*x.shape[0])
NUM_VAL = x.shape[0] - int(0.8*x.shape[0])
x_train = np.zeros((NUM_TRAIN,x.shape[1], x.shape[2], x.shape[3]),dtype=np.float32)
y_train = np.zeros((NUM_TRAIN,y.shape[1], y.shape[2], y.shape[3]),dtype=np.float32)
x_val = np.zeros((NUM_VAL,x.shape[1], x.shape[2], x.shape[3]),dtype=np.float32)
y_val = np.zeros((NUM_VAL,y.shape[1], y.shape[2], y.shape[3]),dtype=np.float32)

np.copyto(x_train,x[:NUM_TRAIN,:,:,:])
np.copyto(y_train,y[:NUM_TRAIN,:,:,:])
np.copyto(x_val,x[NUM_TRAIN:,:,:,:])
np.copyto(y_val,y[NUM_TRAIN:,:,:,:])

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

input_shape = x_train[0].shape

model = my_custom_unet(
    input_shape,
    num_classes=4,
    filters=64,
    output_activation='softmax',
    num_layers=3  
)

model_filename = 'segm_SEGM_'+ str(cv) + '.h5'

callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True
)

model.compile(
    optimizer=Adam(lr=0.0001), 
#    loss = weighted_categorical_crossentropy(weights),
    loss = 'categorical_crossentropy',
    metrics=[iou, iou_thresholded]
)

#model.summary()

train_gen = get_augmented(x_train, y_train, batch_size=2,data_gen_args=data_gen_args)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=300,
    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint]
)
