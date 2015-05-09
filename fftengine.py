# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:31:34 2015

@author: Sarunya
"""

import sys
import numpy as np
import scipy.ndimage

sys.setrecursionlimit(10000)

def normFFT(images_file):
    # apply to array
    img = np.array(images_file)
    #converte image to frequency domain
    #f=np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))))
    f = np.log(np.abs(np.fft.fft2(img)))
    #scaling
    s=(100./f.shape[0],100./f.shape[1])
    
    #normalized frequency domian
    return scipy.ndimage.zoom(f,s,order = 2)

def G(x,mu,s):
    return 1.0/ np.sqrt(2.0*np.pi)*np.exp(((x-mu)**2)/(-2.0*s**2))
    
def getValue(images):
    wd = 8 # theta_range=wd*wd*2
    f = normFFT(images)
    rmax,cmax = f.shape 

    sg = np.zeros((2*wd,wd))
    sg[0:wd,:]=np.log(np.abs(f[rmax-wd:rmax,0:wd]))
    sg[wd:2*wd,:]=np.log(np.abs(f[0:wd,0:wd])) 
    
    return sg.reshape(-1)
    
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