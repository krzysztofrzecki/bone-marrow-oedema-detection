#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from keras.utils import np_utils
import numpy as np
import h5py
import matplotlib.pyplot as plt


# In[2]:


def showOpencvImage(image, isGray=False):
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap = 'gray')
    plt.show()


# In[3]:


class HDF5DatasetGenerator:
    def __init__(self, dbPaths, batchSize, binarize=True, classes=2):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batchSize = batchSize
        self.binarize = binarize
        self.classes = classes

        # open the HDF5 databases for reading and determine the total
        # number of entries in the database
        self.dbs = []
        for dbPath in dbPaths:
            self.dbs.append(h5py.File(dbPath,'r'))
            
        self.numImages = 0
        self.classWeights = np.zeros((2,),dtype = np.float32)

        for db in self.dbs:
            self.numImages += db["images"].shape[0]
            if db["images"].shape[0] > 0:
                if 'weights' in db.keys():
                    self.classWeights[db["labels"][0]] += np.sum(db["weights"])
                else:
                    self.classWeights[db["labels"][0]] += db["images"].shape[0]
                    
        self.patches = []
        for id,db in enumerate(self.dbs):
            self.patches += [(id,n) for n in np.arange(0,db["images"].shape[0])]
            
        np.random.shuffle(self.patches)

    def getWeights(self):
        return self.classWeights
    
    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.numImages-self.batchSize, self.batchSize):
                # extract the images and labels from the HDF dataset
                images = []
                labels = []
                weights = []
                for n in range(i,i + self.batchSize):
                    dbID = self.patches[n][0]
                    imID = self.patches[n][1]
                    images.append((self.dbs[dbID])["images"][imID])
                    labels.append((self.dbs[dbID])["labels"][imID])
                    if 'weights' in self.dbs[dbID].keys():
                        weights.append((self.dbs[dbID])["weights"][imID])
                    else:
                        weights.append(1)

                images = np.asarray(images,dtype = np.float32)
                labels = np.asarray(labels,dtype = np.uint8)
                weights = np.asarray(weights,dtype = np.float32)
                
                # check to see if the labels should be binarized
                if self.binarize:
                    labels = np_utils.to_categorical(labels,self.classes)

                # yield a tuple of images and labels
                yield (images, labels, weights)

            # increment the total number of epochs
            epochs += 1

        def close(self):
        # close the database
            self.db.close()


# In[4]:


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


# In[5]:


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


# In[6]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[7]:


import glob

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

#exit()

PATCHES_DIR = '/home/user/Spine/Wyniki_v1/Klasyfikacja/'
#patchesPositiveFileNameMask = 'featuresPositive*.hdf5'
#patchesNegativeFileNameMask = 'featuresNegative*.hdf5'

#patchesPositiveList = glob.glob(PATCHES_DIR+patchesPositiveFileNameMask)
#patchesPositiveList.sort()

#patchesNegativeList = glob.glob(PATCHES_DIR+patchesNegativeFileNameMask)
#patchesNegativeList.sort()

#print(patchesNegativeList)
#print(patchesPositiveList)

patchesPositiveList = []
for fold in range(5):
    for i in range(0,6):
        patchesPositiveList.append(PATCHES_DIR + 'featuresPositive_' + str(fold) + '_' + str(IDs[fold*6+i]) + '_.hdf5')

for patch in patchesPositiveList:
    print(patch)

patchesNegativeList = []
for fold in range(5):
    for i in range(0,6):
        patchesNegativeList.append(PATCHES_DIR + 'featuresNegative_' + str(fold) + '_' + str(IDs[fold*6+i]) + '_.hdf5')

for patch in patchesNegativeList:
    print(patch)

#exit()

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,SGD
import random

numberOfSplits = 5
splitFilesNumber = len(patchesPositiveList)//numberOfSplits
BATCH_SIZE = 128
VAL_NUM = 5

for modelID in range(0,3):
    for fold in range(0,5):

        testPatchesList = patchesPositiveList[fold*splitFilesNumber:(fold+1)*splitFilesNumber] + patchesNegativeList[fold*splitFilesNumber:(fold+1)*splitFilesNumber]
        
        posTrainPatchesList = patchesPositiveList[0:fold*splitFilesNumber] + patchesPositiveList[(fold+1)*splitFilesNumber:] 
        negTrainPatchesList = patchesNegativeList[0:fold*splitFilesNumber] + patchesNegativeList[(fold+1)*splitFilesNumber:]
        
        valPatchesList = posTrainPatchesList[0:VAL_NUM] + negTrainPatchesList[0:VAL_NUM]
        trainPatchesList = posTrainPatchesList[VAL_NUM:] + negTrainPatchesList[VAL_NUM:]

        random.shuffle(valPatchesList)
        random.shuffle(trainPatchesList)
     
        trainGen = HDF5DatasetGenerator(trainPatchesList, BATCH_SIZE, classes=2)
        valGen = HDF5DatasetGenerator(valPatchesList, BATCH_SIZE, classes=2)
        classWeights = 1/trainGen.getWeights()
        w = sum(classWeights)
        classWeights = classWeights/w
        print(classWeights)


        print(trainGen.numImages)
        print(valGen.numImages)


        model = MiniVGGNet.build(width=20, height=20, depth=5,classes=2)
        opt = Adam(lr=1e-3)
        model.compile(loss=weighted_categorical_crossentropy(classWeights),optimizer=opt,metrics=["accuracy"])

        model_filename = 'retrainedModelMV_' + str(fold) + '_' + str(modelID) + '_.h5'

        callback_checkpoint = ModelCheckpoint(
            model_filename, 
            verbose=1, 
            monitor='val_loss', 
            save_best_only=True
        )

        model.fit_generator(trainGen.generator(),
            steps_per_epoch=trainGen.numImages // BATCH_SIZE,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // BATCH_SIZE,
            epochs=75,
            max_queue_size=10,
            callbacks=[callback_checkpoint], verbose=1,class_weight = classWeights)

        del valPatchesList
        del trainPatchesList
        del trainGen
        del valGen
        del model
    


# In[ ]:




