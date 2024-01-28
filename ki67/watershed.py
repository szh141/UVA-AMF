from skimage import io,morphology,measure
from skimage import feature,filters,color
from skimage import segmentation
from skimage.util import invert


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import os
import glob

"""
a1 = io.imread('WT  488 KI67 568 T1A 647 LAMP1004_ch00_SV.jpg')
a2 = color.rgb2gray(a1)
a3 = filters.gaussian(a2,5)
thresh = filters.threshold_otsu(a3)
a4 = a3>thresh
a5 = morphology.remove_small_objects(a4,100)
a6 = a2*a5

peak = feature.peak_local_max(a6, exclude_border=False, min_distance=15,indices=False)
seed = measure.label(peak)
#plt.imshow(a6)
#plt.imshow(morphology.binary_dilation(peak, footprint=np.ones((5,5))),cmap='Reds',alpha = 0.8)

a7 = invert(a6)
labels = segmentation.watershed(a7,markers=seed,mask=a5) 

labels.max()

plt.imshow(labels)

#peak = feature.peak_local_max(a6, exclude_border=False, min_distance=15)
#plt.imshow(a6)
#plt.scatter(peak[:,1],peak[:,0],marker="o",color = 'red',s=5)
#plt.show()
"""



def count_DAPI (img1,object_size = 100):
    a1 = io.imread(img1)
    a2 = color.rgb2gray(a1)
    a3 = filters.gaussian(a2,5)
    thresh = filters.threshold_otsu(a3)
    a4 = a3>thresh
    a5 = morphology.remove_small_objects(a4,object_size)
    a6 = a2*a5

    peak = feature.peak_local_max(a6, exclude_border=False, min_distance=15,indices=False)
    seed = measure.label(peak)
    #plt.imshow(a6)
    #plt.imshow(morphology.binary_dilation(peak, footprint=np.ones((5,5))),cmap='Reds',alpha = 0.8)

    a7 = invert(a6)
    labels = segmentation.watershed(a7,markers=seed,mask=a5) 
    
    #peak = feature.peak_local_max(a6, exclude_border=False, min_distance=15)
    #plt.imshow(a6)
    #plt.scatter(peak[:,1],peak[:,0],marker="o",color = 'red',s=5)
    #plt.show()

    return labels.max()


def export(names,DAPI_count):
    cwd = os.getcwd()
    sample= os.path.basename(cwd)
    file_name = f'{sample} DAPI.csv'
    results = pd.DataFrame(list(zip(names, DAPI_count)), columns =['names', 'DAPI_count']) 
    results.to_csv(file_name, index=False)  


extension = '*.jpg'
DAPI_img = glob.glob(f'DAPI/{extension}')    
DAPI_img.sort()                  

names = []
DAPI_count = []

"""
a1 = io.imread(file_list[0])
a2 = count_DAPI(a1)
"""

for file_name in DAPI_img:
    test = count_DAPI(file_name)
    DAPI_count.append(test)
    names.append(file_name)
    
export(names,DAPI_count)