import os
import glob
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam,SGD
from keras.models import load_model
import argparse

   
class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(16, (3, 3), padding="same",input_shape=inputShape,kernel_regularizer=regularizers.l1(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), padding="same",kernel_regularizer=regularizers.l1(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same",kernel_regularizer=regularizers.l1(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same",kernel_regularizer=regularizers.l1(0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model


def readBinaryData(n,SIZE,H,nbytes,BO='BE'):

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
                    if BO =='BE':
                        a = 256*byte[0] + byte[1]
                    elif BO == 'LE':
                        a = byte[0] + 256*byte[1]
                        
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

import os

def descriptor(caseClass,R,C,S,data,mask,BOXXY=10,BOXZ=2):
    MIN = -5
    descriptor = []
    descriptor.append(caseClass)
    for s in range(-BOXZ,BOXZ+1):
        for r in range(-BOXXY,BOXXY):
            for c in range(-BOXXY,BOXXY):
                if R+r>=0 and R+r<data.shape[0] and C+c>=0 and C+c<data.shape[1] and S+s>=0 and S+s<data.shape[2]:
                    descriptor.append(data[R+r,C+c,S+s])
                else:
                    descriptor.append(MIN)
    return descriptor


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


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



LEFT_BONE_TOP = 60
LEFT_BONE_BOTTOM = 70
RIGHT_BONE_TOP = 80
RIGHT_BONE_BOTTOM = 90
LEFT_MIDDLE_BONE_TOP = 100
LEFT_MIDDLE_BONE_BOTTOM = 110
RIGHT_MIDDLE_BONE_TOP = 120
RIGHT_MIDDLE_BONE_BOTTOM = 130

REFERENCE = 50
JOINT_LINE = 200


ap = argparse.ArgumentParser()

ap.add_argument("-c", "--cv", required=True,help="fold id")
ap.add_argument("-f", "--fileID", required=True,help="file id")

args = vars(ap.parse_args())

import random
SEED = 10

import random
SEED = 10
PATCHES_DIR = '/home/user/Spine/Wyniki_v0/Klasyfikacja/Iteration1/Patches_MV/'
patchesFileNameMask = 'features*.hdf5'
patchesList = glob.glob(PATCHES_DIR+patchesFileNameMask)
patchesList.sort()
random.seed(SEED)
random.shuffle(patchesList)

IDs = [f.split('/')[-1].split('_')[1] for f in patchesList]

for patch in IDs:
    print(patch)

MODEL_DIR = '/home/user/Spine/Wyniki_v1/Klasyfikacja/'

REG_DIR = '/home/user/Spine/Wyniki_v1/Segmentacja/MyUnet/AfterStep6_PredRegions/'
STIR_DIR = '/home/user/Spine/Data/STIR/'
REG1_DIR = '/home/user/Spine/Data/Regions/'

numberOfSplits = 5
splitFilesNumber = len(patchesList)//numberOfSplits
MARGIN = 5
BOXXY = 10
BOXZ = 2

fold = int(args['cv'])
fileID = int(args['fileID'])

#for fold in range(0,5):

IDs = IDs[fold*splitFilesNumber:(fold+1)*splitFilesNumber]
valSTIRs = [file for file in glob.glob(STIR_DIR+'STIR*.raw') if file.split('/')[-1].split('_')[1] in IDs]
valSTIRs.sort()
valREGs = [file for file in glob.glob(REG_DIR+'Region*.raw') if file.split('/')[-1].split('_')[1] in IDs]
valREGs.sort()
valREGs1 = [file for file in glob.glob(REG1_DIR+'Region*.raw') if file.split('/')[-1].split('_')[1] in IDs]
valREGs1.sort()

print('split',len(valSTIRs),len(valREGs))

for patch in IDs:
    print(patch)

for patch in valSTIRs:
    print(patch)

for patch in valREGs:
    print(patch)

#exit()

classWeights = np.ones((2,),dtype = np.float32)

models = []
for modelID in range(0,3):
    model_filename = MODEL_DIR + 'retrainedModelMV_' + str(fold) + '_' + str(modelID) + '_.h5'
    model = load_model(model_filename,custom_objects = {'loss':weighted_categorical_crossentropy(classWeights)})
    models.append(model)

for numer, (s,r,r1) in enumerate(zip(valSTIRs,valREGs,valREGs1)):

    if numer != fileID:
        continue

    name_s = s.split('/')[-1].split('.')[0]
    _ , num_s,SIZE_s,_,H_s,B_s,_ = name_s.split('_')
    stir = readBinaryData(s,int(SIZE_s),int(H_s),int(B_s))

    name_r = r.split('/')[-1].split('.')[0]
    _ , num_r,SIZE_r,_,H_r,B_r,_ = name_r.split('_')
    reg = readBinaryData(r,int(SIZE_r),int(H_r),int(B_r))

    name_r1 = r1.split('/')[-1].split('.')[0]
    _ , num_r1,SIZE_r1,_,H_r1,B_r1,_ = name_r1.split('_')
    reg1 = readBinaryData(r1,int(SIZE_r1),int(H_r1),int(B_r1))

    SIZE = min(int(SIZE_s),int(SIZE_r))

    if min(int(SIZE_s),int(SIZE_r)) != max(int(SIZE_s),int(SIZE_r)):
        print(name_s,' error')
        break

    H = min(int(H_s),int(H_r))
    stir = stir[:,:,:H]
    reg = reg[:,:,:H]
    reg1 = reg1[:,:,:H]

    mean = np.mean(stir[reg==REFERENCE])
    std = np.std(stir[reg==REFERENCE])
    stir = (stir-mean)/std

    lesions = np.zeros((int(SIZE_s),int(SIZE_s),H),dtype=np.uint8)

    cases = [(R,C,S) for R in range(0,SIZE) for C in range(0,SIZE) for S in range(0,H) if ((reg[R,C,S]>=LEFT_BONE_TOP and reg[R,C,S]<=RIGHT_MIDDLE_BONE_BOTTOM) or (reg1[R,C,S]>=LEFT_BONE_TOP and reg1[R,C,S]<=RIGHT_MIDDLE_BONE_BOTTOM))]

    print(len(cases))
    
    for id,case in enumerate(cases):
        if id%1000==0:
            print(id)
        R,C,S = case
        features = descriptor(0,int(R),int(C),int(S),stir,reg,BOXXY=BOXXY,BOXZ=BOXZ)[1:]
        image = np.zeros((2*BOXZ+1,2*BOXXY,2*BOXXY),dtype = np.float64)
        np.copyto(image,np.reshape(features,(2*BOXZ+1,2*BOXXY,2*BOXXY)))
        image = np.swapaxes(image,0,1)
        image = np.swapaxes(image,1,2)
        image = np.expand_dims(image,0)

        pred = 0
        for model in models:
            pred += np.argmax(model.predict(image))

        if pred>=1:
            lesions[R,C,S] = 255
            
    name = MODEL_DIR + '_'.join(['Recognitions',num_s,SIZE_s,SIZE_s,str(H),'1_.raw'])
    writeBinaryData(lesions,name)






