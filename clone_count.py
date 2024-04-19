# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:39:34 2024

@author: shao
"""

#pip install -U scikit-learn

import napari
from skimage import img_as_float,data
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

#remove the "Detailed" lines from csv first
df = pd.read_csv('p28d.csv')

cols = [3,4,5,6,7,8]
df1 = df.drop(df.columns[cols],axis=1)
#df1.head()
#df1.count()

points = np.array(df1)

viewer = napari.Viewer()
points_layer = viewer.add_points(
    points,
    size = 50
)


# P28 has eps of 75 and min_samples of 15
db = DBSCAN(eps=75, min_samples=5).fit(df1)
labels = db.labels_
print(labels.max())


"""
point_properties = {
    #'confidence': np.array(img_as_float(model.labels_)),
    'confidence': np.array(img_as_float(labels)),

}

prediction_layer = viewer.add_points(
    points,
    properties=point_properties,
    face_color='confidence',
    face_colormap='inferno',
    size = 50
)
"""

points_2 = points[labels == -1]
points_3 = points[labels != -1]

#visualize the clusters
negative_layer = viewer.add_points(
    points_2,
    size = 50)

labels_clusters = labels[labels!=-1]
point_properties = {
    #'confidence': np.array(img_as_float(model.labels_)),
    'confidence': np.array(img_as_float(labels_clusters)),
}

clusters_layer = viewer.add_points(
    points_3,
    properties=point_properties,
    face_color='confidence',
    face_colormap='inferno',
    size = 50)


"""
labels.shape
(labels ==-1).sum()

from collections import Counter
a = Counter(labels)
print(a)
"""

from collections import Counter
a = Counter(labels)
pd.DataFrame(a,index=[0]).to_csv("Clusters_count_min15.csv")


"""
test = pd.read_csv('Clusters_count.csv')
test2 = test[test>10]
test2.head()

row_nan_count = test2.isna().sum(axis=1)
row_nan_count
"""
