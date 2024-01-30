#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:11:37 2024

@author: chongmingwang
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


#use pretrained model, 2D fluorescence
from stardist.models import StarDist2D
model = StarDist2D.from_pretrained('2D_versatile_fluo')

from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
#from csbdeep.utils import normalize
import matplotlib.pyplot as plt

img = test_image_nuclei_2d()

#labels, _ = model.predict_instances(normalize(img))
labels, _ = model.predict_instances(img)


plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")