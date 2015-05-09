# -*- coding: utf-8 -*-
"""
Created on Sat May 02 21:57:16 2015

@author: Sarunya
"""

import os
import sys
import numpy as np
from PIL import Image
import time
import skimage.feature as ft

sys.setrecursionlimit(10000)

def timestamp(tt=time.time()):
    st=time.time()    
    print("    took: %.2f sec"%(st-tt))
    return st

def normLBP(images_file):
    METHOD = 'uniform'
    R = 1 #radius
    P = 8 #n_points
    img = np.array(images_file)
    f = ft.local_binary_pattern(img, P, R, METHOD)

    
    return f

def getValue(images):
    f = normLBP(images)

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