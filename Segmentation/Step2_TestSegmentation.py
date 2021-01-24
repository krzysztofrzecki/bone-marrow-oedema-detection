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

def writeBinaryData(d,n):

    f=open(n,"wb")
    for i in range(0,d.shape[2]):
        for j in range(0,d.shape[0]):
            for k in range(0,d.shape[0]):
                byte = f.write(d[j,k,i])
    f.close()

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

def iou1(y_true, y_pred, level,smooth=0.1):

    y_true_f = np.zeros(y_true.shape,dtype=np.uint8)
    y_pred_f = np.zeros(y_true.shape,dtype=np.uint8)
    
    y_true_f[y_true==level] = 1
    y_pred_f[y_pred==level] = 1
    
#    intersection = np.sum(np.multiply(y_true_f, y_pred_f))
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)



config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

SEED = 10
LES_DIR = '/home/user/Spine/Data/GroundTruth_MajorityVoting/'
T1_DIR = '/home/user/Spine/Data/T1/'
LAB_DIR = '/home/user/Spine/Data/BoneLabels/'
SAVE_DIR = './'

numbers = [name.split('_')[2] for name in glob.glob(LES_DIR + 'MV*.raw')]

random.seed(SEED)
random.shuffle(numbers)
step = len(numbers)//5

for cv in range(0,5):

    st = cv*step
    en = st + step

    t1_dir_list = [n for n in glob.glob(T1_DIR+'T1*.raw') if n.split('_')[1] in numbers[st:en]]
    t1_dir_list.sort()    
   
    input_shape = (512,512,1)

    model = my_custom_unet(
        input_shape,
        num_classes=4,
        filters=64,
        output_activation='softmax',
        num_layers=3  
    )

    model.compile(
        optimizer=Adam(lr=0.0001), 
    #    loss = weighted_categorical_crossentropy(weights),
        loss = 'categorical_crossentropy',
        metrics=[iou, iou_thresholded]
    )

    model_filename = 'segm_SEGM_'+str(cv)+'.h5'

    model.load_weights(model_filename)
    
    size = (512,512)
    for t in t1_dir_list:

        name_t = t.split('/')[-1].split('.')[0]
        _ , num_t,SIZE_t,_,H_t,B_t,_ = name_t.split('_')

        t1 = readBinaryData(t,int(SIZE_t),int(H_t),int(B_t))
        FAC_T1 = 255./ np.max(t1)

        imgs_list = []
        for s in range(0,int(H_t)):
            im = cv2.resize(t1[:,:,s]*FAC_T1,size)
            imgs_list.append(im)
   
        imgs_np = np.asarray(imgs_list)

        x_val = np.asarray(imgs_np, dtype=np.float32)/255
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)

        print("x_val: ", x_val.shape)

        pred3D = np.zeros(t1.shape,dtype=np.uint8)
        
        for n in range(0,x_val.shape[0]):
            y_pred = model.predict(x_val[n:n+1])
            pred = np.zeros(y_pred.shape[1:3],dtype=np.uint8)
            pred = np.argmax(y_pred[0,:,:,:],-1)
            pred[pred !=0] = 255
            pred = cv2.resize(pred,(t1.shape[0],t1.shape[1]),interpolation=cv2.INTER_NEAREST)
            pred = np.asarray(pred,dtype=np.uint8)
            np.copyto(pred3D[:,:,n],pred)

        writeBinaryData(pred3D,SAVE_DIR+'PredBoneLabels_'+num_t + '_' + SIZE_t + '_' + SIZE_t + '_' + H_t + '_1_.raw')


