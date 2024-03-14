# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:02:22 2024

@author: shao
"""

#pip install imaris-ims-file-reader
#pip install skan
from skimage import filters,morphology,measure


from imaris_ims_file_reader.ims import ims
a = ims('Python code.ims') # folder and file name

a.shape
a_image = a[0,:,:,:,:]
#not isotropic voxel yet


a_small = a_image[0]
a_big = a_image[1]
a_big.max()


import napari
#viewer = napari.Viewer()
viewer = napari.view_image(a_big)

mask_small = a_small>0

mask_labeled = measure.label(mask_small)
new_layer = viewer.add_image(mask_labeled)



from skimage.morphology import skeletonize
skeleton = skeletonize(mask_small)
skel_labeled = measure.label(skeleton)
new_layer = viewer.add_image(skel_labeled)

props = measure.regionprops_table(skel_labeled,intensity_image=a_big,
                                  properties=['label','area','intensity_max'])

import pandas as pd
df = pd.DataFrame(props)
df.head()

len(pd.unique(df['intensity_max']))


props2 = measure.regionprops_table(mask_labeled,intensity_image=a_big,
                                  properties=['label','area','intensity_max'])

df2 = pd.DataFrame(props2)
df2.head()

len(pd.unique(df2['intensity_max']))