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


import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images",
        bufSize=1000):
        # check to see if the output path exists, and if so, raise
        # an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied `outputPath` already "
                "exists and cannot be overwritten. Manually delete "
                "the file before continuing.", outputPath)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims,dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],),dtype="int")
        self.weights = self.db.create_dataset("weights",(dims[0],),dtype="float")

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": [],"weights": []}
        self.idx = 0

    def add(self, rows, labels,weights):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        self.buffer["weights"].extend(weights)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.weights[self.idx:i] = self.buffer["weights"]
        self.idx = i
        self.buffer = {"data": [], "labels": [],"weights": []}

    def storeClassLabels(self, classLabels):
        # create a dataset to store the actual class label names,
        # then store the class labels
        dt = h5py.special_dtype(vlen=str) # `vlen=unicode` for Py2.7
        labelSet = self.db.create_dataset("label_names",(len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()


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



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import random


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
SEED = 10


ap = argparse.ArgumentParser()

ap.add_argument("-c", "--cv", required=True,help="fold id")
ap.add_argument("-f", "--fileID", required=True,help="file id")

args = vars(ap.parse_args())


PATCHES_DIR = '/home/user/Spine/Wyniki_v0/Klasyfikacja/Iteration1/Patches_MV/'
patchesFileNameMask = 'features*.hdf5'

patchesList = glob.glob(PATCHES_DIR+patchesFileNameMask)
patchesList.sort()

random.seed(SEED)
random.shuffle(patchesList)

for patch in patchesList:
    print(patch)

#print(len(patchesList))


MODEL_DIR = '/home/user/Spine/Wyniki_v1/Klasyfikacja/ScisleWgArtykulu/'

REG_DIR = '/home/user/Spine/Data/Regions/'
STIR_DIR = '/home/user/Spine/Data/STIR/'
LES_DIR = '/home/user/Spine/Data/GroundTruth_MajorityVoting/'


numberOfSplits = 5
splitFilesNumber = len(patchesList)//numberOfSplits
MARGIN = 5
BOXXY = 10
BOXZ = 2
casesP = []
casesN = []

fold = int(args['cv'])
fileID = int(args['fileID'])


################
## ewaluacja tylko na danych treningowych i walidacyjnych!!!
## zmiana w stosunku do artykułu
valPatchesList = patchesList[fold*splitFilesNumber:(fold+1)*splitFilesNumber]
IDs = ['_'+file.split('/')[-1].split('_')[1]+'_' for file in valPatchesList]

valSTIRs = [file for file in glob.glob(STIR_DIR+'STIR*.raw') if '_'+file.split('/')[-1].split('_')[1]+'_' in IDs]
valSTIRs.sort()
valREGs = [file for file in glob.glob(REG_DIR+'Region*.raw') if '_'+file.split('/')[-1].split('_')[1]+'_' in IDs]
valREGs.sort()
valLESs = [file for file in glob.glob(LES_DIR+'MV*.raw') if '_'+file.split('/')[-1].split('_')[1]+'_' in IDs]
valLESs.sort()
print(IDs)

for st in valSTIRs:
    print(st)

for st in valREGs:
    print(st)

for st in valLESs:
    print(st)

print('lists created')

folds = [ f for f in range(0,5) if f != fold]

print(folds)

models = []
for f in folds:
    for modelID in range(0,3):
        model = MiniVGGNet.build(width=20, height=20, depth=5,classes=2)
        opt = Adam(lr=1e-3)
        model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
        model_filename = MODEL_DIR + 'initialModelMV_' + str(f) + '_' + str(modelID) + '_.h5'
        model = load_model(model_filename)
        models.append(model)

print('model read',len(models))
################

for numer, (s,r,l) in enumerate(zip(valSTIRs,valREGs,valLESs)):

    if numer != fileID:
        continue

    name_s = s.split('/')[-1].split('.')[0]
    _ , num_s,SIZE_s,_,H_s,B_s,_ = name_s.split('_')
    stir = readBinaryData(s,int(SIZE_s),int(H_s),int(B_s))

    name_r = r.split('/')[-1].split('.')[0]
    _ , num_r,SIZE_r,_,H_r,B_r,_ = name_r.split('_')
    reg = readBinaryData(r,int(SIZE_r),int(H_r),int(B_r))

    name_l = l.split('/')[-1].split('.')[0]
    _ , num_l,SIZE_l,_,H_l,B_l,_ = name_l.split('_')
    les = readBinaryData(l,int(SIZE_l),int(H_l),int(B_l))
    
    SIZE = min(int(SIZE_s),int(SIZE_r),int(SIZE_l))

    if min(int(SIZE_s),int(SIZE_r),int(SIZE_l)) != max(int(SIZE_s),int(SIZE_r),int(SIZE_l)):
        print(name_s,' error')
        break

    H = min(int(H_s),int(H_r),int(H_l))
    stir = stir[:,:,:H]
    reg = reg[:,:,:H]
    les = les[:,:,:H]

    mean = np.mean(stir[reg==REFERENCE])
    std = np.std(stir[reg==REFERENCE])
    stir = (stir-mean)/std

    print("start")

#################################################
#           Creatig positive features           #
#################################################
    pos = [(R,C,S) for R in range(0,SIZE) for C in range(0,SIZE) for S in range(0,H) if les[R,C,S] and reg[R,C,S]>=LEFT_BONE_TOP and reg[R,C,S]!=JOINT_LINE]
    descriptors = []
    weights = []
    for id,case in enumerate(pos):
        if id%100==0:
            print('create positive descriptors',id,len(pos))
        R,C,S = case
        descriptors.append(descriptor(1,int(R),int(C),int(S),stir,reg,BOXXY=BOXXY,BOXZ=BOXZ))
        weights.append(1)
        
    for caseID,case in enumerate(pos):        #increase weight of difficult cases
        if caseID%100==0:
            print('testing positive models',caseID,len(pos))
        R,C,S = case
        features = descriptor(1,int(R),int(C),int(S),stir,reg,BOXXY=BOXXY,BOXZ=BOXZ)
        image = np.zeros((2*BOXZ+1,2*BOXXY,2*BOXXY),dtype = np.float64)
        np.copyto(image,np.reshape(features[1:],(2*BOXZ+1,2*BOXXY,2*BOXXY)))
        image = np.swapaxes(image,0,1)
        image = np.swapaxes(image,1,2)
        image = np.expand_dims(image,0)
        preds = []
        for model in models:
            preds.append(np.argmax(model.predict(image)))
        weights[caseID] += len(models)-np.sum(preds)
            
    np.random.shuffle(descriptors)
    
    outputPath = 'featuresPositive_' + str(fold) + '_' + num_l + '_.hdf5'
    writer = HDF5DatasetWriter((len(descriptors),2*BOXXY,2*BOXXY,2*BOXZ+1), outputPath)
    for caseID,desc in enumerate(descriptors):
        if caseID%100==0:
            print('writing positive descriptors',caseID,len(descriptors))
        label = desc[0]
        features = desc[1:]
        weight = weights[caseID]
        image = np.zeros((2*BOXZ+1,2*BOXXY,2*BOXXY),dtype = np.float64)
        np.copyto(image,np.reshape(features,(2*BOXZ+1,2*BOXXY,2*BOXXY)))
        image = np.swapaxes(image,0,1)
        image = np.swapaxes(image,1,2)
        writer.add([image], [label],[weight])
        del image
    writer.close()
    
    casesPos = len(descriptors)
    del descriptors
    del weights
    del pos

    print('end positive')
#################################################

#################################################
#           Creatig negative features           #
#################################################
    dilated = cv2.dilate(les,None,iterations=MARGIN)
    descriptors = []
    usedCases = []
    weights = []
    neg = [(R,C,S) for R in range(0,SIZE) for C in range(0,SIZE) for S in range(0,H) if dilated[R,C,S]==0 and reg[R,C,S]>=LEFT_BONE_TOP and reg[R,C,S]<=RIGHT_MIDDLE_BONE_BOTTOM]
    for id,case in enumerate(neg):        #increase weight of difficult cases
        if id%100 == 0:
            print('testing negative models',id,len(neg))
        R,C,S = case
        features = descriptor(0,int(R),int(C),int(S),stir,reg,BOXXY=BOXXY,BOXZ=BOXZ)
        image = np.zeros((2*BOXZ+1,2*BOXXY,2*BOXXY),dtype = np.float64)
        np.copyto(image,np.reshape(features[1:],(2*BOXZ+1,2*BOXXY,2*BOXXY)))
        image = np.swapaxes(image,0,1)
        image = np.swapaxes(image,1,2)
        image = np.expand_dims(image,0)
        preds = []
        for model in models:
            preds.append(np.argmax(model.predict(image)))
        if np.sum(preds):
            usedCases.append(((R,C,S)))
            descriptors.append(features)
            weights.append(np.sum(preds))
    
    np.random.shuffle(neg)
    for id,case in enumerate(neg):
        if len(descriptors) >= casesPos:
            break
        if case not in usedCases:
            if id%100 == 0:
                print('create negative descriptors',id,len(neg))
            R,C,S = case
            descriptors.append(descriptor(0,int(R),int(C),int(S),stir,reg,BOXXY=BOXXY,BOXZ=BOXZ))
            usedCases.append(case)
            weights.append(1)
    
    np.random.shuffle(descriptors)
    outputPath = 'featuresNegative_' + str(fold) + '_' +  num_l + '_.hdf5'
    writer = HDF5DatasetWriter((len(descriptors),2*BOXXY,2*BOXXY,2*BOXZ+1), outputPath)
    for caseID,desc in enumerate(descriptors):
        if caseID%100 == 0:
            print('writing negative descriptors',caseID,len(descriptors))
        label = desc[0]
        features = desc[1:]
        weight = weights[caseID]
        image = np.zeros((2*BOXZ+1,2*BOXXY,2*BOXXY),dtype = np.float64)
        np.copyto(image,np.reshape(features,(2*BOXZ+1,2*BOXXY,2*BOXXY)))
        image = np.swapaxes(image,0,1)
        image = np.swapaxes(image,1,2)
        writer.add([image], [label],[weight])
        del image
    writer.close()
    
    casesNeg = len(descriptors)
    del descriptors
    del weights
    del neg
    
    casesP.append(casesPos)
    casesN.append(casesNeg)                

    print('end')

del valPatchesList
del models

print(casesPos,casesNeg)

f = open('cases.dat','a+')
for cp,cn in zip(casesP,casesN):
    f.write(str(cp)+' ' + str(cn)+'\n')
f.close()  





