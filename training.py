import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
import matplotlib.pyplot as plt
from turtle import color
from sklearn import metrics
from sklearn.model_selection import learning_curve

from plot import plot_training
from callbacks import tensor_board, callbacks
from models_generators import createModel, fitModel
from config import EPOCHS, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE


train_samples=118
test_samples=40

train_set = "points/train"
test_set = "points/test"
val_set = "data/val"

#Loading train set data as image generator - rescaling 1./255
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(directory=train_set, 
                                                                        class_mode='binary', 
                                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                        color_mode="grayscale",
                                                                        batch_size=BATCH_SIZE,
                                                                        shuffle=True)

#Loading test set data as image generator - rescaling 1./255
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(directory=test_set, 
                                                                         class_mode='binary', 
                                                                         target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                         color_mode="grayscale",
                                                                         batch_size=BATCH_SIZE,
                                                                         shuffle=True)

#Model learning 
history = fitModel(createModel(IMG_WIDTH, IMG_HEIGHT), trainGen=train_generator, epoch=EPOCHS, stepsPE=train_samples//BATCH_SIZE,
 validationGen=test_generator, stepsVal=test_samples//BATCH_SIZE)

#training for arrays?

#train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# fitModel(createModel(), trainGen=train_generator.flow(train_tensor, train_label, batch_size=BATCH_SIZE, subset='training'),
#  epoch=EPOCHS, stepsPE=100, validationGen=validation_generator.flow(val_tensor, val_label, batch_size=BATCH_SIZE, subset='validation'), 
#  stepsVal=15)

plot_training(history=history)