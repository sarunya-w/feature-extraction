# -*- coding: utf-8 -*-
"""
Created on Fri May 01 02:10:31 2015

@author: Sarunya
"""

import sys
import numpy as np
from matplotlib import pyplot as plt
#from cv2 import HOGDescriptor
from skimage.feature import hog

sys.setrecursionlimit(10000)


clmax = 11 #clmax is amount of class

def normHOG(images_file):
    img = np.array(images_file)
    width, height = img.shape
    # SKIMAGE    
    f = hog(img, normalise=True,pixels_per_cell=(height//4, width//4))
    
    # OPENCV HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins
    #opencv_hog = HOGDescriptor((200,200), (16,16), (8,8), (8,8), 9)
    #f = opencv_hog.compute(img)

    return f 

def getValue(images):
    f = normHOG(images)
    return f.reshape(-1)

def getVector(images_files,class_files,samples,isTrain):
    from PIL import Image
    bs = 200
    clmax = 11 #clmax is amount of class
    sub_img = []
    sub_cs = []
    bb = bs//2
    pos = []
    
    for f in xrange(len(images_files)):
        img = Image.open(images_files[f]).convert('L')
        w , h = img.size
        pixels=[]
        for i in xrange(samples):
            r = np.random.randint(bb, h-bb)
            c = np.random.randint(bb, w-bb)
            pixels.append((c,r))
            if isTrain==False:
                pos.append((c,r))
            box = (c-bb, r-bb, c + bb, r + bb)
            output_img = img.crop(box)
            sub_img.append(getValue(output_img))
    
        if isTrain:
            cimg = Image.open(class_files[f]).convert('L')
            for p in pixels:   
                sub_cs.append(cimg.getpixel(p))
        
    if isTrain:
        sub_img=np.array(sub_img,dtype=np.float32)
        sub_cs=np.array(sub_cs,dtype=np.uint32)
        sub_cs[sub_cs==255]= clmax - 1
    else:
        sub_cs=None

    return (sub_img ,sub_cs,pos)
