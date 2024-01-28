from skimage import io,morphology,measure,color,filters
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import os
import glob

def read_img (file_path):
    img1 = io.imread(file_path)
    img2 = color.rgb2gray(img1)
    img3 = img2[10:-100,10:-10]
    
    return img3

def seg (img1,hole_size):
    thresh = filters.threshold_otsu(img1)
    img2 = img1<thresh
    img3 = morphology.remove_small_holes(img2,hole_size)
    
    return img3

def radius(img1):
    distance = ndi.distance_transform_edt(img1)
    distance_masked = distance>3
    skeleton = morphology.skeletonize(img1)
    skeleton_masked = skeleton*distance_masked.astype(int)
    img2 = distance*skeleton_masked
    
    return img2

def diameter(img1):
    diameter = img1[np.nonzero(img1)]*2
    (n1,bins,patches) = plt.hist(diameter,bins=np.arange(0,54,6))
    n2 = n1/np.sum(n1)
    n3 = n2[1:] 
    n3[0] += n2[0] #sum all diameters below 12 pixels or 6um
  
    return n3

def export(frequency):
    cwd = os.getcwd()
    sample= os.path.basename(cwd)
    file_name = f'frequency_{sample}.csv'
    df = pd.DataFrame(frequency)
    df.to_csv(file_name, index=False)

            
            
# pixel_size = 0.5 um measured by Fiji

extension = '*.jpg'
file_list = glob.glob(extension)                         

frequency = []

for file_name in file_list:
    test = read_img(file_name)
    test2 = seg(test,300)
    test3 = radius(test2)
    test4 = diameter(test3)
    
    frequency.append(test4)
plt.show()

export(frequency)

#frequency_KI = []
#frequency_KI = frequency
#frequency_WT = []
#frequency_WT = frequency

xrange = np.arange(4.5,25.5,3) #plot bin center
fig = plt.plot()
for i in range(len(frequency)):
    plt.scatter(xrange,frequency[i],marker = 'o',label=f'{i+1} ki')
    plt.ylim(0,0.5)
    plt.xticks(np.arange(6,24,3)) # converted pixel to um
    #plt.legend()

"""

for i in range(len(frequency)):
    plt.scatter(xrange,frequency[i],marker = 'v',label=f'{i+1} WT')
    plt.ylim(0,0.5)
    plt.xticks(np.arange(6,24,3)) # converted pixel to um
    #plt.legend()
"""


plt.savefig(f'{sample}.png')
