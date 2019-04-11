#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:40:46 2019

Based on Zhengyou Zhang's 1999 paper on "A Flexible New Technique for Camera
Calibration" - Microsoft Reserach.

https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf


@author: kartikmadhira
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize as opt
import glob
import os 


#get V matrix from the corners obtained from the chessboard corners list and the vij matrix.
def getV(corners):
    # corners - corners in order obtained from the cv2.findChessboardCorners() function.
    reqPoints=np.zeros((1,2))
    reqPoints[0,:]=corners[0]
    reqPoints=np.vstack((reqPoints,corners[8]))
    reqPoints=np.vstack((reqPoints,corners[53]))
    reqPoints=np.vstack((reqPoints,corners[45]))
    
    worldPoints=np.array([[21.5,21.5],[21.5*9,21.5],[21.5*9,21.5*6],[21.5,21.5*6]])
    #because the srcPoints are the worldCoordinates or the plane coors
    H=cv2.findHomography(worldPoints,reqPoints)
    H=H[0]
    #print(H)
    v12=getVIJMatrix(H,0,1)
    v11=getVIJMatrix(H,0,0)
    v22=getVIJMatrix(H,1,1)
    V=np.array([v12.transpose(),(v11-v22).transpose()])
    #getting the right eigenvector 
    #diag=np.linalg.svd(np.matmul(V.transpose(),V))    
    return V

def getNewProj(H,K):
    
    K_inv = np.linalg.inv(K)
    B=np.matmul(K_inv,H)
    
    #this is to ensure that the if the determinant is negative, it means that the ray is from behind the camera.
    #We want a projected ray from the front of the camera.
    if(np.linalg.det(B)<0):
        B=B*(-1)
        print('negative det')
    
    b1=B[0:3,0:1]
    b2=B[0:3,1:2]
    b3=B[0:3,2:3]
    
    l=(np.linalg.norm(b1,ord=2)+np.linalg.norm(b2,ord=2))/2
    
    l=1/l
    Rt=np.zeros([3,4])
    Rt[0:3,0:1]=l*b1
    Rt[0:3,1:2]=l*b2
    Rt[0:3,2:3]=np.cross(Rt[0:3,0:1],Rt[0:3,1:2],axis=0)
    Rt[0:3,3:4]=l*b3
    P=np.matmul(K,Rt)
   # P=Rt
    return P,Rt

#calculate the vij matrix where i is the column and j is the row
def getVIJMatrix(H,i,j):
    vij=np.array([(H[0,i]*H[0,j]),(H[0,i]*H[1,j])+(H[1,i]*H[0,j]),(H[1,i]*H[1,j]),(H[2,i]*H[0,j])+(H[0,i]*H[2,j]),
         (H[2,i]*H[1,j])+(H[1,i]*H[2,j]),H[2,i]*H[2,j]])
    return vij
    

#set of parameters that needs to be optimized for nonlinear optimization
def parameters(A,k1,k2):
    
    #A  - initial estimate of the instrinsic parameter
       
    a1 = np.reshape(np.array([A[0][0],A[0][1],A[0][2],A[1][1],A[1][2]]),(5,1))
    a3 = np.reshape(np.array([k1,k2]),(2,1))
    param = np.concatenate([a3,a1])
    return param



#minimization function for nonlinear optimizartion.
def minimizeFunc(params,worldL,actualPoints,rtStack,N):
    projectedPoints=np.zeros((702,2))
    A=np.array([[params[2],0,params[4]],
                  [0,params[5],params[6]],
                  [0,0,1]])
    K = np.reshape(params[0:2],(2,1))
    k=-1  
    for i in range(N):
        if(i%54==0):
            k+=1
            rt=rtStack[k]
            #print(k)
        Rt=np.reshape(rt,(3,4))
        P=np.matmul(A,Rt)

        points=np.matmul(P,np.array([[worldL[i][0]],[worldL[i][1]],[0],[1]]))
        points=points/points[2]
        normPoints = np.matmul(Rt,np.array([[worldL[i][0]],[worldL[i][1]],[0],[1]]))
        normPoints = normPoints/normPoints[2]

        uHat = points[0] + (points[0] - A[0][2])*[(K[0]*((normPoints[0])**2 + (normPoints[1])**2)) +
                                                        (K[1]*((normPoints[0])**2 + (normPoints[1])**2)**2)]
        vHat = points[1] + (points[1] - A[1][2])*[(K[0]*((normPoints[0])**2 + (normPoints[1])**2)) +
                                                        (K[1]*((normPoints[0])**2 + (normPoints[1])**2)**2)]
               
        projectedPoints[i][0]=uHat
        projectedPoints[i][1]=vHat

    error=np.linalg.norm(np.subtract(projectedPoints,actualPoints),axis=1)**2
    error=np.reshape(error,(702,))

    return error


def main():
    
    files=glob.glob("Calibration_Imgs/*.jpg")
    
    #obtain the V matrix by stacking each V matrix from the calibrating images.
    vStack=np.zeros((1,6))
    for eachImage in files:
        img=cv2.imread(eachImage)
        gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,(9,6),None)
        if (ret):
            V=getV(corners)
            vStack=np.vstack((vStack,V))

    vStack=vStack[1:len(vStack)]
    #apply SVD decomposition to this V matrix.
    diag=np.linalg.svd(np.matmul(vStack.transpose(),vStack)) 
    
    #B is now the eigenvector corresponding to least singular value.
    B=diag[2][5]
    B11=B[0]
    B12=B[1]
    B13=B[3]
    B22=B[2]
    B23=B[4]
    B33=B[5]
    v0=(B12*B13-B11*B23)/((B11*B22)-B12**2)
    lam=B33-(((B13**2)+v0*(B12*B13-B11*B23))/B11)
    alpha=math.sqrt(lam*1/B11)
    beta=math.sqrt(lam*B11/((B11*B22)-B12**2))
    gamma=-B12*(alpha**2)*beta/lam
    u0=(gamma*v0/beta)-((B13*(alpha**2))/lam)
    K=np.array([[alpha ,gamma, u0 ],[0,beta,v0],[0,0,1]])
    print("intial intrinsic matrix is ",K)
    
    #loop over to get the world coordinates of the chessboard corners which are 21.5 mm in size     
    board_dims = (9,6)
    world = []
    sq_dim = 21.5
    for col in range(0, board_dims[1]):
        for row in range(0, board_dims[0]):
            world.append([row*sq_dim,col*sq_dim])
    world = np.array(world)
    world=world+21.5
    worldL=np.zeros((1,2))
    for i in range(13):
        worldL=np.vstack((worldL,world))
        #print(np.shape(world))
    worldL=worldL[1:]
    worldL.shape        
    actualPoints=np.zeros((1,2))
    rtStack=np.zeros((1,12))
        
    #loop over each of the image to get the Rt or the pose of the camera wrt to each of the image.
    #and then stack them to set them as initial values to feed into the nonlinear optimzer.
    for eachImage in files:
        img=cv2.imread(eachImage)
        gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,(9,6),None)
        corners=np.reshape(corners,(54,2))
        actualPoints=np.vstack((actualPoints,corners))
        reqPoints=np.zeros((1,2))
        reqPoints[0,:]=corners[0]
        reqPoints=np.vstack((reqPoints,corners[8]))
        reqPoints=np.vstack((reqPoints,corners[53]))
        reqPoints=np.vstack((reqPoints,corners[45]))
        
        worldPoints=np.array([[21.5,21.5],[21.5*9,21.5],[21.5*9,21.5*6],[21.5,21.5*6]])
        #because the srcPoints are the worldCoordinates or the plane coors
        H=cv2.findHomography(worldPoints,reqPoints)
        H=H[0]
        _,Rt=getNewProj(H,K)
        Rt=np.reshape(Rt,(1,12))
        rtStack=np.vstack((rtStack,Rt))
            
    actualPoints=actualPoints[1:]
    rtStack=rtStack[1:]

    #setting the initial parameters to feed into the optimizer
    params=parameters(K,0,0)
    N=702
    
    #nonlinear optimizer with Levenberg-Marquardt optimizer.
    K_prime = opt.least_squares(fun=minimizeFunc, x0=np.squeeze(params), method="lm",args=(worldL,actualPoints,rtStack,N) )
    res=K_prime
    
    #get the optimzed parameters.
    A = np.array([[res.x[2],res.x[3],res.x[4]],
                  [0,res.x[5],res.x[6]],
                  [0,0,1]])
    K = np.reshape(res.x[0:2],(2,1))
    print("New A",A)
    print("Distortion parameters",K)
    print("Cost",res.cost)
    
    #set a distortion matrix to undistort images
    
    distortion = np.array([K[0],K[1],0,0,0],dtype=float)
    files=glob.glob("Calibration_Imgs/*.jpg")
    rectCorners=np.zeros((1,2))
    unrectCorners=np.zeros((1,2))
    files=glob.glob("Calibration_Imgs/*.jpg")
    if not os.path.exists('Rectified'):
        os.makedirs('Rectified')
    
    
    #loop over images to get the corner values from the findChessboardCorners function and the projection from world
    #coordinates to the image plane. RMS to be obtained in the end.
    for i,eachImage in enumerate(files):
        imgCal=cv2.imread(eachImage)
        imgGray=cv2.cvtColor(imgCal,cv2.COLOR_BGR2GRAY)
        undist = cv2.undistort(imgCal,A,distortion)
        #undist=cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
        ret,cornersList = cv2.findChessboardCorners(undist,(9,6),None)
       
        cv2.drawChessboardCorners(undist,(9,6),cornersList,True)
        
        cv2.imwrite('Rectified/'+str(i)+'.jpg',undist)

        ret,corners=cv2.findChessboardCorners(imgGray,(9,6),None)
        corners=np.reshape(corners,(54,2))
        actualPoints=np.vstack((actualPoints,corners))
        reqPoints=np.zeros((1,2))
        reqPoints[0,:]=corners[0]
        reqPoints=np.vstack((reqPoints,corners[8]))
        reqPoints=np.vstack((reqPoints,corners[53]))
        reqPoints=np.vstack((reqPoints,corners[45]))
        
        worldPoints=np.array([[21.5,21.5],[21.5*9,21.5],[21.5*9,21.5*6],[21.5,21.5*6]])
        #because the srcPoints are the worldCoordinates or the plane coors
        H=cv2.findHomography(worldPoints,reqPoints)
        H=H[0]
        _,Rt=getNewProj(H,A)
        P=np.matmul(A,Rt)
    

        #this loops over to find the image points by introducing the distortion coefficents
        #points - The coordinates obtained by using the full projection matrix - IMAGE PLANE POINTS
        #normPoints - The coordinates obtained by using just Rt and world coorindates - CAMERA COORDINATES POINTS

        for j in range(54):
            points=np.matmul(P,np.array([[worldL[j][0]],[worldL[j][1]],[0],[1]]))
            #print(points.shape)
            points=points/points[2]
            normPoints = np.matmul(Rt,np.array([[worldL[i][0]],[worldL[i][1]],[0],[1]]))
            normPoints= normPoints/normPoints[2]
    
            uHat = points[0] + (points[0] - A[0][2])*[(K[0]*((normPoints[0])**2 + (normPoints[1])**2)) +
                                                            (K[1]*((normPoints[0])**2 + (normPoints[1])**2)**2)]
            vHat = points[1] + (points[1] - A[1][2])*[(K[0]*((normPoints[0])**2 + (normPoints[1])**2)) +
                                                            (K[1]*((normPoints[0])**2 + (normPoints[1])**2)**2)]
            
            point=np.array([uHat,vHat])
            point=np.reshape(point,(1,2))
            rectCorners=np.vstack((rectCorners,point))
        unrectCorners=np.vstack((unrectCorners,corners))
    rectCorners=rectCorners[1:]
    unrectCorners=unrectCorners[1:]
    rms=np.mean(np.linalg.norm(rectCorners-unrectCorners,axis=1))
    print("RMS value is: ",rms)
    
     
if __name__ == '__main__':
    main()

    

