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


def count_GFP (img1,img2,object_size = 100):
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
    
    b1 = io.imread(img2)
    b2 = color.rgb2gray(b1)  
    #value = filters.threshold_multiotsu(b2)
    #b3 = b2>value[1]
    b3 = b2>0.45
    
    masked_labels = labels*b3
    return len(np.unique(masked_labels))


def export(names,GFP_count):
    cwd = os.getcwd()
    sample= os.path.basename(cwd)
    file_name = f'{sample} GFP.csv'
    results = pd.DataFrame(list(zip(names, GFP_count)), columns =['names', 'GFP_count']) 
    results.to_csv(file_name, index=False)  


extension = '*.jpg'
DAPI_img = glob.glob(f'DAPI/{extension}')
DAPI_img.sort()                      
GFP_img = glob.glob(f'GFP/{extension}')  
GFP_img.sort()  

names = []
GFP_count = []


for DAPI,GFP in zip(DAPI_img,GFP_img):
    test = count_GFP(DAPI,GFP)
    GFP_count.append(test)
    names.append(DAPI)
    #break
    
export(names,GFP_count)

for DAPI,GFP in zip(DAPI_img,GFP_img):
    print(DAPI,GFP)

    