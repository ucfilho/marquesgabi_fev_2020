import numpy as np
import cv2
import pandas as pd
import mahotas.features.texture as mht
import mahotas.features

from scipy.signal import find_peaks
from scipy.signal import peak_prominences
from scipy.signal import peak_widths

def Mahotas(ww,Size,Sub_Size,Crop,img_name):  # ww represent the photo in question
  #row=len(img_name)
  row=1 # unica foto escolhida
  col=Size*Size
  pw=np.zeros((row,col))
  for kk in range(row):
    pw[kk,:]=ww.ravel()
    p_foto=pw[kk,:].reshape(Size,Size)
    GLCM=[]
    glcm_haralick=[]
    x_ref=[]
    Count=Sub_Size
    p=np.zeros((Sub_Size,Sub_Size))
    j_ref=0
    Cada_foto=[]
    Posicao_X=[]
    Posicao_Y=[]
    for k in range(Size):
      if((k+Sub_Size-1)<Size):
        for i in range(Sub_Size):
          Posicao_X.append(Crop+i)
          for j in range(Sub_Size):
            p[i,j]=p_foto[Crop+i,j+k]
            Posicao_Y.append(j+k)

        WT=np.copy(p) 
        Cada_foto.append(WT.ravel())
        x_ref.append(Count-Sub_Size)
        Count=Count+1
        
        Nomes=['ASM','constrast','correl','variance','inv diff mom','sum aveg',
               'sum var','sum entropy','entropy','dif var','dif entropy','IMC1',
               'IMC2']

        Mahotas =pd.DataFrame(mahotas.features.haralick(p.astype(int)), columns =Nomes)

        GLCM=[]
        for ii in Nomes:
          GLCM.append(Mahotas[ii].mean())
          #print('cheguei aqui!!!')
        glcm_haralick.append(GLCM)
  df=pd.DataFrame(glcm_haralick,columns=Nomes)
  return df
