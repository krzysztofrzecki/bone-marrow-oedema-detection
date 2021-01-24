#https://igraph.org/python/doc/tutorial/install.html#
#dokumentacja - https://igraph.org/python/doc/python-igraph.pdf

#!sudo apt install build-essential python-dev libxml2 libxml2-dev zlib1g-dev
#!pip install igraph
#!conda install -c conda-forge python-igraph 
# https://networkx.github.io/documentation/latest/_downloads/networkx_reference.pdf


import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def showOpencvImage(image, isGray=False):
    fig = plt.figure(figsize=(6, 6))
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




def components3D(t):
    
    dum = np.zeros(t.shape,dtype=np.uint32)
    label = 0
    
    for x in range(0,t.shape[0]):
        for y in range(0,t.shape[1]):
            for z in range(0,t.shape[2]):
                if t[x,y,z] and dum[x,y,z]==0:
                    label = label+1
                    dum[x,y,z] = label
                    lista = []
                    lista.append((x,y,z))
                    while len(lista):
                        X,Y,Z = lista[0]
                        lista.pop(0)
                        for x1 in range(-1,2):
                            for y1 in range(-1,2):
                                for z1 in range(-1,2):
                                    if X+x1>=0 and X+x1<t.shape[0] and Y+y1>=0 and Y+y1<t.shape[1] and Z+z1>=0 and Z+z1<t.shape[2]:
                                        if t[X+x1,Y+y1,Z+z1] and dum[X+x1,Y+y1,Z+z1]==0:
                                            dum[X+x1,Y+y1,Z+z1] = label
                                            lista.append((X+x1,Y+y1,Z+z1))
    return label,dum




def findCut(a,START,END):
    from igraph import Graph

    g = Graph(directed=True)

    N = a.shape[0]*a.shape[1]*a.shape[2]
    g.add_vertices(N+2)

    lista = []
    weights = []
    for k in range(1,a.shape[2]-1):
        for i in range(1,a.shape[0]-1):
            for j in range(1,a.shape[1]-1):
                n = j + i*a.shape[1] + k*a.shape[0]*a.shape[1]
                lista.append((n,n+1))
                lista.append((n,n-1))
                lista.append((n,n+a.shape[0]))
                lista.append((n,n-a.shape[0]))
                lista.append((n,n+a.shape[0]*a.shape[1]))
                lista.append((n,n-a.shape[0]*a.shape[1]))
                weights.append(int(a[i,j,k]))
                weights.append(int(a[i,j,k]))
                weights.append(int(a[i,j,k]))
                weights.append(int(a[i,j,k]))
                weights.append(int(a[i,j,k]))
                weights.append(int(a[i,j,k]))

    for k in range(0,a.shape[2]):
        for i in range(0,a.shape[0]):
            n = START + i*a.shape[1]  + k*a.shape[0]*a.shape[1]
            lista.append((N,n))
            weights.append(10000)

    for k in range(0,a.shape[2]):
        for i in range(0,a.shape[0]):
            n = a.shape[0]*i + END +  k*a.shape[0]*a.shape[1]
            lista.append((n,N+1))
            weights.append(10000)

    for k in range(0,a.shape[2]):
        for i in range(0,a.shape[0]):
            n = a.shape[0]*i + k*a.shape[0]*a.shape[1]
            lista.append((n,n+1))
            weights.append(10000)

    for k in range(0,a.shape[2]):
        for i in range(0,a.shape[0]):
            n = a.shape[0]*i + a.shape[0]-1+ k*a.shape[0]*a.shape[1]
            lista.append((n,n-1))
            weights.append(10000)

    for k in range(0,a.shape[2]):
        for j in range(0,a.shape[1]):
            n = j+ k*a.shape[0]*a.shape[1]
            lista.append((n,n+a.shape[1]))
            weights.append(10000)

    for k in range(0,a.shape[2]):
        for i in range(0,a.shape[0]):
            n = a.shape[1]*a.shape[0]-1-i+ k*a.shape[0]*a.shape[1]
            lista.append((n,n-a.shape[0]))
            weights.append(10000)

    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[1]):
            n = j + i*a.shape[1] 
            lista.append((n,n+a.shape[0]*a.shape[1]))
            weights.append(int(10000))
            n = j + i*a.shape[1] +(a.shape[2]-1)*a.shape[0]*a.shape[1]
            lista.append((n,n-a.shape[0]*a.shape[1]))
            weights.append(int(10000))

    g.add_edges(lista)
    g.es["weight"] = weights[:]
    print(g.is_weighted())

    mf = g.maxflow(N,N+1,g.es["weight"])
    print(mf.value)  
    return mf


class cluster:
    def __init__(self, label,masa):
        self.label = label
        self.masa = masa


# In[ ]:

ap = argparse.ArgumentParser()
ap.add_argument("-m","--numer", required=True,
	help="file id, from 0 to 29 included")
args = vars(ap.parse_args())


PRED_DIR = './MyUnet/'

filenames = glob.glob(PRED_DIR+'Corr1BoneLabels*.raw')
#filenames = [name for name in glob.glob(PRED_DIR +'Corr1BoneLabels*.raw') if int(name.split('/')[-1].split('_')[1]) >=31]
filenames.sort()

print(len(filenames))

size = 400

NNN = int(args["numer"])

for numer,filename in enumerate(filenames):

    if numer !=NNN:
        continue

    print(numer,filename)

    name = filename.split('/')[-1].split('.')[0]
    _ , num,SIZE,_,H,_,_ = name.split('_')
    
    SIZE = int(SIZE)
    H = int(H)
    
    lab = readBinaryData(filename,SIZE,H,1)

    segmOrg = np.zeros((size,size,H),dtype=np.uint8)
    for s in range(0,int(H)):
        im = cv2.resize(lab[:,:,s],(size,size),interpolation=cv2.INTER_NEAREST)
        np.copyto(segmOrg[:,:,s],im)

    segm = np.zeros(segmOrg.shape,dtype=np.uint8)
    np.copyto(segm,segmOrg)
    segm[:,:,0] = 0
    segm[:,:,-1] = 0

    np.copyto(segm,segmOrg)

    while True:
        segm = cv2.dilate(segm,None,iterations=1)
        n,dum = components3D(segm)
        masy = []
        for label in range(1,n+1):
            masy.append(cluster(label,np.sum(dum[dum==label])/label))
        masy = sorted(masy,key=lambda cluster: 1/cluster.masa)
        YMIN = segm.shape[0]
        YMAX = 0
        for x in range(0,segm.shape[0]):
            for y in range(0,segm.shape[1]):
                for z in range(0,segm.shape[2]):
                    if dum[x,y,z]==masy[0].label:
                        if y>YMAX:
                            YMAX=y
                        if y<YMIN:
                            YMIN=y
        print(YMIN,YMAX)
        if YMIN<segm.shape[0]*0.15 and YMAX>segm.shape[0]*0.85:
            np.copyto(segm,segmOrg)
            for l in range(1,len(masy)):
                segm[dum==masy[l].label]=0
    #        for l in range(1,len(masy)):
    #            segm[dum==masy[l].label]=0
            break


    segm[:,:,0] = 0
    segm[:,:,-1] = 0

    np.copyto(segmOrg,segm)


    for k in range(0,segm.shape[2]):
        for i in range(0,segm.shape[0]):
            suma = 0
            for j in range(0,segm.shape[1]//4):
                suma = suma + segm[i,j,k]
            if suma:
                for j in range(0,segm.shape[1]//4):
                    if segm[i,j,k]:
                        break
                    segm[i,j,k] = 255

    for k in range(0,segm.shape[2]):
        for i in range(0,segm.shape[0]):
            suma = 0
            for j in range(segm.shape[1]-1,3*segm.shape[1]//4,-1):
                suma = suma + segm[i,j,k]
            if suma:
                for j in range(segm.shape[1]-1,3*segm.shape[1]//4,-1):
                    if segm[i,j,k]:
                        break
                    segm[i,j,k] = 255


    BOX = int(10*segm.shape[1]/400)
    S = (int)(segm.shape[1]/2-BOX)
    E = (int)(segm.shape[1]/2+BOX)
    segm[:,S:E,1:segm.shape[2]-1]=255

    #for z in range(0,segm.shape[2]):
    #    showOpencvImage(segm[:,:,z])

    START = 0
    END = segm.shape[1]//2
    mfLeft = findCut(segm,START,END)

    START = segm.shape[1]//2
    END = segm.shape[1]-1
    mfRight = findCut(segm,START,END)

    b = np.zeros(segmOrg.shape,dtype=np.uint8)
    np.copyto(b,segmOrg)

    print(len(mfLeft.partition[0]))
    print(len(mfLeft.partition[1]))

    n = np.asarray(mfLeft.partition[0])
    z = n//(segmOrg.shape[1]*segmOrg.shape[0])
    y = (n - z*segmOrg.shape[1]*segmOrg.shape[0])//segmOrg.shape[0]
    x = n - z*segmOrg.shape[1]*segmOrg.shape[0] - y*segmOrg.shape[0]

    x[x<0] = 0
    x[x>=segmOrg.shape[0]]=0
    y[y<0] = 0
    y[y>=segmOrg.shape[1]]=0
    z[z<0] = 0
    z[z>=segmOrg.shape[2]]=0

    for i in range(0,len(mfLeft.partition[0])):
        if (segmOrg[y[i],x[i],z[i]]):
            b[y[i],x[i],z[i]] = 128 

    n = np.asarray(mfRight.partition[1])
    z = n//(segmOrg.shape[1]*segmOrg.shape[0])
    y = (n - z*segmOrg.shape[1]*segmOrg.shape[0])//segmOrg.shape[0]
    x = n - z*segmOrg.shape[1]*segmOrg.shape[0] - y*segmOrg.shape[0]

    x[x<0] = 0
    x[x>=segmOrg.shape[0]]=0
    y[y<0] = 0
    y[y>=segmOrg.shape[1]]=0
    z[z<0] = 0
    z[z>=segmOrg.shape[2]]=0

    for i in range(0,len(mfRight.partition[1])):
        if (segmOrg[y[i],x[i],z[i]]):
            b[y[i],x[i],z[i]] = 64

#    for z in range(0,b.shape[2]):  
#        showOpencvImage(b[:,:,z])
    b[b==128] = 1
    b[b==255] = 2
    b[b==64] = 3
    name = 'Final_' + str(num) + '_' + str(b.shape[0]) + '_' + str(b.shape[1]) + '_' + str(H) + '_' +'1_.raw'
    writeBinaryData(b,name)    





