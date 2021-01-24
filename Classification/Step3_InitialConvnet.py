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
        for db in self.dbs:
            self.numImages += db["images"].shape[0]
            
        self.patches = []
        for id,db in enumerate(self.dbs):
            self.patches += [(id,n) for n in np.arange(0,db["images"].shape[0])]
            
        np.random.shuffle(self.patches)

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
                for n in range(i,i + self.batchSize):
                    dbID = self.patches[n][0]
                    imID = self.patches[n][1]
                    images.append((self.dbs[dbID])["images"][imID])
                    labels.append((self.dbs[dbID])["labels"][imID])

                images = np.asarray(images,dtype = np.float32)
                labels = np.asarray(labels,dtype = np.uint8)
                
                # check to see if the labels should be binarized
                if self.binarize:
                    labels = np_utils.to_categorical(labels,self.classes)

                # yield a tuple of images and labels
                yield (images, labels)

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


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import glob
import random

SEED = 10
PATCHES_DIR = '/home/user/Spine/Wyniki_v0/Klasyfikacja/Iteration1/Patches_MV/'
patchesFileNameMask = 'features*.hdf5'

patchesList = glob.glob(PATCHES_DIR+patchesFileNameMask)
patchesList.sort()

random.seed(SEED)
random.shuffle(patchesList)

for patch in patchesList:
    print(patch)

print(len(patchesList))

# In[7]:


from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,SGD

numberOfSplits = 5
splitFilesNumber = len(patchesList)//numberOfSplits
BATCH_SIZE = 128
VAL_NUM = 5

for modelID in range(0,3):
    for fold in range(0,5):

        testPatchesList = patchesList[fold*splitFilesNumber:(fold+1)*splitFilesNumber]
        trainPatchesList = patchesList[0:fold*splitFilesNumber] + patchesList[(fold+1)*splitFilesNumber:]

        valPatchesList = trainPatchesList[0:VAL_NUM]
        trainPatchesList = trainPatchesList[VAL_NUM:]


        print(valPatchesList)
        print(trainPatchesList)
        print(testPatchesList)
        
        trainGen = HDF5DatasetGenerator(trainPatchesList, BATCH_SIZE, classes=2)
        valGen = HDF5DatasetGenerator(valPatchesList, BATCH_SIZE, classes=2)

        print(trainGen.numImages)
        print(valGen.numImages)


        model = MiniVGGNet.build(width=20, height=20, depth=5,classes=2)

        opt = Adam(lr=1e-3)
        #opt = SGD(lr=1e-4)

        model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

        model_filename = 'initialModelMV_' + str(fold) + '_' + str(modelID) + '_.h5'

        callback_checkpoint = ModelCheckpoint(
            model_filename, 
            verbose=1, 
            monitor='val_accuracy', 
            save_best_only=True
        )

        model.fit_generator(trainGen.generator(),
            steps_per_epoch=trainGen.numImages // BATCH_SIZE,
            validation_data=valGen.generator(),
            validation_steps=valGen.numImages // BATCH_SIZE,
            epochs=75,
            max_queue_size=10,
            callbacks=[callback_checkpoint], verbose=1)

        del valPatchesList
        del trainPatchesList
        del trainGen
        del valGen
        del model
    


# In[ ]:




