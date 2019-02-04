#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats as st
import skimage.transform
from sklearn.cluster import KMeans
import argparse

def gkern(kernlen, nsig):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def DoG(scales,orient,size):
    orients=np.linspace(0,360,orient)
    kernels=[]
    kernel=gkern(size,scales)
    border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
    sobelx64f = cv2.Sobel(kernel,cv2.CV_64F,1,0,ksize=3, borderType=border)
    plt.subplots(3,5,figsize=(20,20))
    for i,eachOrient in enumerate(orients):
        image=skimage.transform.rotate(sobelx64f,eachOrient)
        plt.subplots_adjust(hspace=1.0,wspace=1.5)
        plt.subplot(3,5,i+1)
        plt.imshow(image,cmap='binary')
        kernels.append(image)
        image=0
    plt.savefig('DoG.png')
    plt.close()
    return kernels

def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

def gaussian1d(sigma, mean, x, ord):
    x = np.array(x)
    x_ = x - mean
    var = sigma**2

    # Gaussian Function
    g1 = (1/np.sqrt(2*np.pi*var))*(np.exp((-1*x_*x_)/(2*var)))

    if ord == 0:
        g = g1
        return g
    elif ord == 1:
        g = -g1*((x_)/(var))
        return g
    else:
        g = g1*(((x_*x_) - var)/(var**2))
        return g

def gaussian2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    return g

def log2d(sup, scales):
    var = scales * scales
    shape = (sup,sup)
    n,m = [(i - 1)/2 for i in shape]
    x,y = np.ogrid[-m:m+1,-n:n+1]
    g = (1/np.sqrt(2*np.pi*var))*np.exp( -(x*x + y*y) / (2*var) )
    h = g*((x*x + y*y) - var)/(var**2)
    return h

def makefilter(scale, phasex, phasey, pts, sup):

    gx = gaussian1d(3*scale, 0, pts[0,...], phasex)
    gy = gaussian1d(scale,   0, pts[1,...], phasey)

    image = gx*gy

    image = np.reshape(image,(sup,sup))
    return image

def makeLMfilters():
    sup     = 49
    scalex  = np.sqrt(2) * np.array([1,2,3])
    norient = 6
    nrotinv = 12

    nbar  = len(scalex)*norient
    nedge = len(scalex)*norient
    nf    = nbar+nedge+nrotinv
    F     = np.zeros([sup,sup,nf])
    hsup  = (sup - 1)/2

    x = [np.arange(-hsup,hsup+1)]
    y = [np.arange(-hsup,hsup+1)]

    [x,y] = np.meshgrid(x,y)

    orgpts = [x.flatten(), y.flatten()]
    orgpts = np.array(orgpts)

    count = 0
    for scale in range(len(scalex)):
        for orient in range(norient):
            angle = (np.pi * orient)/norient
            c = np.cos(angle)
            s = np.sin(angle)
            rotpts = [[c+0,-s+0],[s+0,c+0]]
            rotpts = np.array(rotpts)
            rotpts = np.dot(rotpts,orgpts)
            F[:,:,count] = makefilter(scalex[scale], 0, 1, rotpts, sup)
            F[:,:,count+nedge] = makefilter(scalex[scale], 0, 2, rotpts, sup)
            count = count + 1

    count = nbar+nedge
    scales = np.sqrt(2) * np.array([1,2,3,4])

    for i in range(len(scales)):
        F[:,:,count]   = gaussian2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, scales[i])
        count = count + 1

    for i in range(len(scales)):
        F[:,:,count] = log2d(sup, 3*scales[i])
        count = count + 1

    return F


def saveLMFilters(F):
    for i in range(0,48):
        plt.subplot(4,12,i+1)
        plt.axis('off')
        plt.imshow(F[:,:,i], cmap = 'binary')
    plt.savefig('LM.png')

def saveGaborFilters(G):
    for i in range(0,36):
        plt.subplot(6,6,i+1)
        plt.axis('off')
        plt.imshow(G[i], cmap = 'binary')
    plt.savefig('Gabor.png')


def saveHalfDisks(bank1,bank2):
    for i in range(0,15):
        plt.subplot(5,6,i+1)
        plt.axis('off')
        plt.imshow(bank1[i], cmap = 'binary')
    for i in range(0,15):
        plt.subplot(5,6,16+i)
        plt.axis('off')
        plt.imshow(bank2[i], cmap = 'binary')
       
    plt.savefig('HDMasks.png')
    plt.close()


def half_disk(radius):
    a=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    mask2 = x*x + y*y <= radius**2
    a[mask2] = 0
    b=np.ones((2*radius+1,2*radius+1))
    y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
    p = x>-1 
    q = y>-radius-1
    mask3 = p*q
    b[mask3] = 0

    return a, b


def chiSquareDist(imgMap,bank,flipbank,bins):
    imgMapCopy=imgMap
    for i,eachFilter in enumerate(bank):
        chiSqr=imgMap*0
        for eachBin in range(1,bins):
            #find the id in the list and make a mask
            tmpMask=np.ma.masked_where(imgMap==eachBin,imgMap)
            tmpMask=tmpMask.mask.astype(np.int)
            #conv with left and right images
            #border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
            g=cv2.filter2D(tmpMask,-1,bank[i].astype(np.int))
            h=cv2.filter2D(tmpMask,-1,flipbank[i].astype(np.int))
            tmpVal=((g-h)**2)/(g+h)
            chiSqr=chiSqr+tmpVal
        imgMapCopy=np.dstack((imgMapCopy,0.5*chiSqr))
    return imgMapCopy
    




def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--indexPic',dest='indexPic',type=int, default=1, help='input for the index of the image')
    Parser.add_argument('--imgPath',dest='imgPath', default='/home/kartikmadhira/CMSC733/YourDirectoryID_hw0/Phase1/BSDS500/Images/', help='Path to load images from, Default:BasePath')
    Args = Parser.parse_args()
    indexPic = Args.indexPic
    imgPath=Args.imgPath	
    imgPath=imgPath+str(indexPic)+'.jpg'
    
    
    """
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
    dog1=DoG(16,15,49)
	
    """
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
    
    F = makeLMfilters()
    saveLMFilters(F)
	
    
    """
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
    
    angle1=np.linspace(0,360,12)
    gaborKernels=[]
    gabor1=gabor_fn(9,0.25,1,1,1)
    for eachAngle in angle1:
        gab1=skimage.transform.rotate(gabor1,eachAngle)
        gaborKernels.append(gab1)
    angle2=np.linspace(0,360,12)
    gabor2=gabor_fn(16,0.25,1,1,1)
    for eachAngle in angle2:
        gab2=skimage.transform.rotate(gabor2,eachAngle)
        gaborKernels.append(gab2)
    angle3=np.linspace(0,360,12)
    gabor3=gabor_fn(16,0.25,1,1,1)
    for eachAngle in angle3:
        gab3=skimage.transform.rotate(gabor3,eachAngle)
        gaborKernels.append(gab3)
    
    saveGaborFilters(gaborKernels)
    
    
	
    """
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
    a1,b1=half_disk(25)
    orient=np.linspace(0,360,15)
    halfDiskBank1=[]
    for eachAngle in orient:
        rotatedMask=skimage.transform.rotate(b1,eachAngle)
        image1=np.logical_or(a1,rotatedMask).astype(np.int)
        halfDiskBank1.append(image1)
        image1=0
    halfDiskBank2=[]
    for each in halfDiskBank1:
        image=np.flip(each).astype(np.int)
        halfDiskBank2.append(image)
        image=0

    saveHalfDisks(halfDiskBank1,halfDiskBank2)
	
    """
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""
  
    ss=cv2.imread(imgPath,0)
    w,h=ss.shape
    ss2=ss
    for i in range(48):
        border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
        image=cv2.filter2D(ss,-1,F[:,:,i],borderType=border)
        ss2=np.dstack((ss2,image))
        image=0
    for i in range(15):
        border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
        image=cv2.filter2D(ss,-1,dog1[i],borderType=border)
        ss2=np.dstack((ss2,image))
        image=0
    for i in range(12):
        border = cv2.borderInterpolate(0, 1, cv2.BORDER_CONSTANT)
        image=cv2.filter2D(ss,-1,gaborKernels[i],borderType=border)
        ss2=np.dstack((ss2,image))
        image=0
        
    _,_,d=ss2.shape
    ss3=ss2[:,:,1:d]
    ss4=np.reshape(ss3,((w*h),d-1))
    ss4.shape
    
    """
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
    kmeans = KMeans(n_clusters=64, random_state=2)
    kmeans.fit(ss4)
    labels = kmeans.predict(ss4)
    textonMap=np.reshape(labels,(w,h))
    plt.imshow(textonMap)
    plt.savefig('TextonMap_'+str(indexPic)+'.png')
    plt.close()
    print('saved DOG and LM')    

    """
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
    
    Tg=chiSquareDist(textonMap,halfDiskBank1,halfDiskBank2,64)
    Tg=Tg[:,:,1:16]
    plt.imshow(np.mean(Tg,axis=2))
    plt.savefig('Tg_'+str(indexPic)+'.png')
    plt.close()
    print('saved DOG and LM')  
    
    """
	Generate Brightness Map
	Perform brightness binning 
    """
    
    newSSshape=np.reshape(ss,((w*h),1))
    kmeansBrightness = KMeans(n_clusters=16, random_state=2)
    kmeansBrightness.fit(newSSshape)
    
    labelsBrightnesss=kmeansBrightness.predict(newSSshape)
    brightMap=np.reshape(labelsBrightnesss,((w,h)))
    plt.imshow(brightMap)
    plt.savefig('BrightMap_'+str(indexPic)+'.png')
    plt.close()
    """
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
    Bg=chiSquareDist(brightMap,halfDiskBank1,halfDiskBank2,16)
    Bg=Bg[:,:,1:16]
    plt.imshow(np.mean(Bg,axis=2))
    plt.savefig('Bg_'+str(indexPic)+'.png')
    plt.close()
    print('saved DOG and LM')  

    

#   
#    
#	"""
#	Generate Color Map
#	Perform color binning or clustering
#	"""
    ssColor=cv2.imread(imgPath)
    ssColor=np.reshape(ssColor,((w*h),3))
    kmeansColor = KMeans(n_clusters=16, random_state=2)
    kmeansColor.fit(ssColor)
    colorMap=kmeansColor.predict(ssColor)
    colorMap=np.reshape(colorMap,(w,h))
    plt.imshow(colorMap)
    plt.savefig('ColorMap_'+str(indexPic)+'.png')
    plt.close()

    
    """
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
    Cg=chiSquareDist(colorMap,halfDiskBank1,halfDiskBank2,16)
    Cg=Cg[:,:,1:16]
    plt.imshow(np.mean(Cg,axis=2))
    plt.savefig('Cg_'+str(indexPic)+'.png')
    plt.close()
    
    """
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
    sobelImagePath='/home/kartikmadhira/CMSC733/YourDirectoryID_hw0/Phase1/BSDS500/SobelBaseline/'+str(indexPic)+'.png'
    sobelBaseline=cv2.imread(sobelImagePath,0)
    
    """
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
    cannyImagePath='/home/kartikmadhira/CMSC733/YourDirectoryID_hw0/Phase1/BSDS500/CannyBaseline/'+str(indexPic)+'.png'
    cannyBaseline=cv2.imread(cannyImagePath,0)
    
    tmp1=(Tg+Bg+Cg)/3
    average=np.mean(tmp1,axis=2)
    
    """
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
    final=np.multiply(average,(0.5*cannyBaseline+0.5*sobelBaseline))
    plt.imshow(final,cmap='binary')
    cv2.imwrite('PbLite_'+str(indexPic)+'.png',final)
    print('saved DOG and LM')  
    plt.close()

#
#	
#
#	
    
if __name__ == '__main__':
    main()
 


