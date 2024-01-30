# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 06:01:14 2024

@author: shao
"""

"""

conda create -n "stardist-gpu-py3-10" python=3.10.0
#tensorflow 2.11.0, python version 3.7-3.10, CUDA 11.2
#Laurent Thomas

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
#install the cudatoolkit package and cudnn

python -m pip install "tensorflow<2.11"
#install tf 2.10

import tensorflow as tf
print(tf.version.VERSION)
print(tf.config.list_physical_devices('GPU'))

#tf 2.x.x
pip install stardist

"""


#use pretrained model, 2D fluorescence
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

from skimage import io,util,color
"""
img1 = test_image_nuclei_2d()
img1.dtype
"""

img = io.imread('WT  488 KI67 568 T1A 647 LAMP1001_ch00_SV.jpg')
img.dtype
img = color.rgb2gray(img)
img.shape
img.max()
normalize(img).max()
"""
# no need to resize to test image (512,512)
img = util.img_as_uint(img)
img.dtype
from skimage.transform import resize
resize(img, (512, 512)).shape
"""

labels, _ = model.predict_instances(normalize(img))


plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")



"""

?normalize
?render_label

labels.shape
plt.imshow(labels)
import numpy as np
print(np.unique(labels))

from skimage import measure
import pandas as pd

prop = measure.regionprops_table(labels,properties=('label','area'))

df = pd.DataFrame(prop)
df.head()

df
df2 = df[df['area']>100] #watershed.py threshold on 100
df2
# a difference in KI between 1077 vs 1126
# a difference in WT between 592 vs 616

"""



