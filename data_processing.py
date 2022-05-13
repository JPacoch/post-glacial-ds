from cProfile import label
from pyproj import transform
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
import matplotlib.pyplot as plt

#Loading prepared point dataset
points = gpd.read_file("C:/Users/pacoc/Desktop/a.shp")
print(points)

#DEM / Hillshade for Poland
src = rasterio.open('C:/Users/pacoc/Desktop/Warsztat/Studia/mag/data/DEM.tif')

windows_list = []

#Get window values
for point in points['geometry']:
    x = point.xy[0][0]
    y = point.xy[1][0]
    row, col = src.index(x,y)
    rst = src.read(1, window=rasterio.windows.Window(col_off=col, row_off=row,
                                                     width=128, height=128))
    windows_list.append(rst)

# print(windows_list[0])
# print(len(windows_list[0]))
# print(type(windows_list[0]))

#Labelling
labels_list=[]
for elem in range(0, len(windows_list)):
    elem = 0
    labels_list.append(elem)

labels_list = np.asarray(labels_list).astype('float32')
print(labels_list)
print(type(labels_list))

# labels_list = tf.keras.utils.to_categorical(labels_list)
# print(labels_list)
# print(type(labels_list))

#List of arrays to list of tensors
#Tensor of tensors
def arrayListToTensor(list):
        tensor_list = tf.convert_to_tensor(list, dtype=tf.float32)
        print(type(tensor_list))
        print(tensor_list.shape)
        print(tensor_list.ndim)
        print(tensor_list.dtype)
        return tensor_list

#List of tensors
# def arrayListToTensor(list):
#     tensor_list=[]
#     for arr in list:
#         arr = tf.convert_to_tensor(arr, dtype=tf.float32)
#         tensor_list.append(arr)
#     print(tensor_list)
#     # print(tensor_list[0].ndim)
#     return tensor_list

tensor = arrayListToTensor(windows_list)
plt.imshow(tensor[1])
plt.show()