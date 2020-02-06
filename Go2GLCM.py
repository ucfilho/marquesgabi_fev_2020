# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import os 
import cv2
import zipfile
from skimage import data
from skimage import filters
from skimage import exposure
import skimage.feature as sk
import numpy as np
import pandas as pd
from google.colab import files
from sklearn.metrics.cluster import entropy
from sklearn.metrics.cluster import homogeneity_score
from numpy import linalg as LA

def GLCM(p):
  k=0
  GLCM=np.zeros((1, 14))  
  nrow,ncol=p.shape
  Size=nrow  #Size=nrow-1
  Nrow=Size
  Ncol=Size
  G=Ncol-1
  Pdif=[] 
  Psom=[]
  Soma=0
  px=[]
  py=[]

  for i in range(Size):
    for j in range(Size):
      Soma=Soma+p[i,j]
  
  for i in range(Size):
    for j in range(Size):
      p[i,j]=p[i,j]/Soma

  for i in range(Size):
    Somax=0
    for j in range(Size):
      Somax=Somax+p[i,j]
    px.append(Somax)

  for j in range(Size):
    Somay=0
    for i in range(Size):
      Somay=Somay+p[i,j]
    py.append(Somay)

  for kr in range(2*G+1):
    Soma_soma=0
    for i in range(Nrow):
      for j in range(Ncol):
        if((i+j)==kr):
          Soma_soma=Soma_soma+p[i,j]
    Psom.append(Soma_soma)

  for kr in range(G+1):
    Soma_dif=0
    for i in range(Nrow):
      for j in range(Ncol):
        if((abs(i-j))==kr):
          Soma_dif=Soma_dif+p[i,j]
    Pdif.append(Soma_dif)

  HXY=0
  HXY1=0
  HXY2=0
  for i in range(Nrow):
    for j in range(Ncol):
      HXY=HXY-p[i,j]*np.log(p[i,j])
      HXY1=HXY1-p[i,j]*np.log(px[i]*py[j])
      HXY2=HXY2-px[i]*py[j]*np.log(px[i]*py[j])
  HX=0
  for i in range(Nrow):
    HX=HX-px[i]*np.log(px[i])
  HY=0
  for j in range(Nrow):
    HY=HY-py[j]*np.log(py[j])
  if(HX>HY):
    maxH=HX
  else:
    maxH=HY

  for i in range(Nrow):
    for j in range(Ncol):
      #print("i=%d j=%d k=%d"%(i,j,k))
      GLCM[k,0]=GLCM[k,0]+p[i,j]**2 #segundo momento angular
      GLCM[k,1]=GLCM[k,1]+(i-j)**2*p[i,j] #contraste


      MIx=0;MIy=0
      for ii in range(Nrow):
        for jj in range(Ncol):
          MIy=MIy+jj*p[ii,jj]
          MIx=MIx+ii*p[ii,jj]
      Sx2=0;Sy2=0;
      for ii in range(Nrow):
        for jj in range(Ncol):
          Sx2=Sx2+(ii-MIx)**2*p[ii,jj]
          Sy2=Sy2+(jj-MIy)**2*p[ii,jj]
      Sx=Sx2**0.5
      Sy=Sy2**0.5


      GLCM[k,2]=GLCM[k,2]+((i*j)*p[i,j]-MIx*MIy)/(Sx*Sy) #Correlacao
      # CONFERIR CORRELACAO EM OUTROS

      GLCM[k,3]=GLCM[k,3]+(i-np.mean(p))**2*p[i,j] #Variancia

      GLCM[k,4]=GLCM[k,4]+(1/(1+(i-j)**2))*p[i,j]#INVERSA DIF MOMENTO
      
    f8=0
      
    for kr in range(len(Psom)):
      GLCM[k,5]=GLCM[k,5]+kr*Psom[kr] # conferir !!!!! ok???
      f8=f8-Psom[kr]*np.log(Psom[kr])

    for kr in range(len(Psom)):
      GLCM[k,6]=GLCM[k,6]+(kr-f8)**2*Psom[kr] # sum of variance

    GLCM[k,7]=f8  # sum entropy
    GLCM[k,8]=GLCM[k,8]-p[i,j]*np.log(p[i,j]) # entropia

    for kr in range(len(Pdif)):
      GLCM[k,9]=GLCM[k,9]+(kr)**2*Pdif[kr] # variance difference
      GLCM[k,10]=GLCM[k,10]-Pdif[kr]*np.log(Pdif[kr]) # difference entropy
      
    GLCM[k,11]=(HXY-HXY1)/maxH # information measure of correlation 1
    GLCM[k,12]=(1-np.exp(-2*(HXY2-HXY)))**0.5 # information measure of correlation 2

    Q=np.zeros((Nrow,Ncol))
    for i in range(Nrow):
      for j in range(Ncol):
        for kr in range(Ncol):
          Q[i,j]=Q[i,j]+p[i,kr]*p[j,kr]/(px[i]*py[kr])
    
    #v = LA.eig(Q)
    v=LA.eigvals(Q)
    n=len(v)
    v.sort()
    #print(v)
    #print(v[n-2])
    GLCM[k,13]=v[n-2]**0.5 # maximal correlation coefficient
  return GLCM
