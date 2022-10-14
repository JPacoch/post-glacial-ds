import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.model_selection

from re import X
from PIL import Image
from cProfile import label
from pyproj import transform
from sklearn.model_selection import train_test_split

from config import PATH_TO_DENUDED, PATH_TO_HILLSHADE, PATH_TO_NONDENUDED

#Loading prepared point dataset
# points = gpd.read_file("C:/Users/pacoc/Desktop/Warsztat/Studia/mag/data/punkty/punkty1.gpkg", layer = 'punkty1')

#DEM / Hillshade for Poland
# src = rasterio.open('C:/Users/pacoc/Desktop/Warsztat/Studia/mag/data/hillshade.tif')

def prepare_array_dirs():
    folders = [f"arrays/denuded", f"arrays/nondenuded"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"{folder} created")

prepare_array_dirs()

def clip_windows(file, type):
    points = gpd.read_file(file)
    src = rasterio.open(PATH_TO_HILLSHADE) #raster
    i = 0
    for point in points["geometry"]:
        x = point.xy[0][0]
        y = point.xy[1][0]
        row, col = src.index(x,y)
        rst = src.read(1, window=rasterio.windows.Window(col_off=col, row_off=row,
                                                        width=128, height=128))
        # plt.imshow(rst, cmap='binary')
        # plt.show()
        clip = Image.fromarray(rst)
        clip.save(f"arrays/{type}/array{i}.png", format="PNG")
        i += 1


clip_windows(PATH_TO_NONDENUDED, "nondenuded")
clip_windows(PATH_TO_DENUDED, "denuded")

def prepare_dirs(dir):
    folders = [f"{dir}/train/denuded", f"{dir}/train/nondenuded", f"{dir}/test/denuded", f"{dir}/test/nondenuded"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"{folder} created")

prepare_dirs("points")

def read_arrays(type):
    list = []
    for file in os.listdir(os.getcwd() + f"\\arrays\\{type}"):
        print(os.path.join(os.getcwd() + f"\\arrays\\", file))
        img = Image.open(os.path.join(os.getcwd() + f"\\arrays\\{type}", file))
        arr = np.array(img)
        list.append(arr)
    return list

def split_and_save(list, type, ratio):
    assert ratio > 0 and ratio < 1, "Ratio must be in range (0,1)"
    assert type in ["denuded", "nondenuded"], "Type must be denuded or nondenuded"
    #assert all(np.shape(i) == (128,128) for i in list), "All array must be of size (128, 128)"
    train, test, train_idx, test_idx = train_test_split(list, np.array(range(0, len(list))), train_size = ratio)
    print(f"{len(train)} / {len(test)}")

    for set, name, idxs in zip([train, test], ["train", "test"], [train_idx, test_idx]):
        path = os.getcwd() + "\\points\\" + name
        for arr, idx in zip(set, idxs):
            img = Image.fromarray(arr)
            img.save(path + f"\\{type}\\{type}{idx}.png", format='PNG')

list = read_arrays("denuded")
print(list)
split_and_save(list, "denuded", 0.75)

list = read_arrays("nondenuded")
split_and_save(list, "nondenuded", 0.75)



####Earlier approach

# windows_list = []

# def createImgDataset(data, class_name, train_test_split=False):
#     print(class_name)

#     #Get window values
#     for point in data['geometry']:
#         x = point.xy[0][0]
#         y = point.xy[1][0]
#         row, col = src.index(x,y)
#         rst = src.read(1, window=rasterio.windows.Window(col_off=col, row_off=row,
#                                                         width=128, height=128))
#         windows_list.append(rst)
#         # plt.imshow(rst, cmap='binary')
#         # plt.show()

#     # print(windows_list[0])
#     # print(len(windows_list[0]))
#     # print(type(windows_list[0]))

#     #Labelling
#     labels_list=[]
#     for elem in range(0, len(windows_list)):
#         elem = 0
#         labels_list.append(elem)

#     labels_list = np.asarray(labels_list).astype('float32')
#     print(labels_list)
#     print(type(labels_list))

#     # labels_list = tf.keras.utils.to_categorical(labels_list)
#     # print(labels_list)
#     # print(type(labels_list))

#     #List of arrays to list of tensors
#     #Tensor of tensors
#     def arrayListToTensor(list):
#             tensor_list = tf.convert_to_tensor(list, dtype=tf.float32)
#             print(type(tensor_list))
#             # print(tensor_list.shape)
#             # print(tensor_list.ndim)
#             # print(tensor_list.dtype)
#             return tensor_list

#     #List of tensors
#     # def arrayListToTensor(list):
#     #     tensor_list=[]
#     #     for arr in list:
#     #         arr = tf.convert_to_tensor(arr, dtype=tf.float32)
#     #         tensor_list.append(arr)
#     #     print(tensor_list)
#     #     # print(tensor_list[0].ndim)
#     #     return tensor_list

#     tensor = arrayListToTensor(windows_list)
#     # plt.imshow(tensor[1])
#     # plt.show()

#     if train_test_split == True:
#         X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(windows_list,
#                                                             labels_list,
#                                                             test_size=0.20,
#                                                             random_state=42)
#         i = 0
#         for arr in range(1,len(X_train)):
#             arr = Image.fromarray(X_train[i])
#             arr.save(f'data/train/{class_name}/{i}.png', format='PNG')
#             i = i + 1

#         j = 0
#         for arr in range(1,len(X_val)):
#             arr = Image.fromarray(X_val[j])
#             arr.save(f'data/val/{class_name}/{j}.png', format='PNG')
#             j = j + 1
#     else:
#         i = 0
#         for arr in range(1,len(windows_list)):
#             arr = Image.fromarray(windows_list[i])
#             arr.save(f'data/manual/{class_name}/{i}.png', format='PNG')
#             i = i + 1

# createImgDataset(points, 'denuded')