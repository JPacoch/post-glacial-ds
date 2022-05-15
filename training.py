import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import learning_curve

from callbacks import tensor_board, callbacks

EPOCHS = 200
BATCH_SIZE = 20
IMG_WIDTH, IMG_HEIGHT = 128, 128
train_samples=78
validation_samples=20

train_set = "data/train"
test_set = "data/test"
val_set = "data/val"

#Loading train set data as image generator - rescaling 1./255
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(directory=train_set, 
                                                                        class_mode='binary', 
                                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                        color_mode="grayscale",
                                                                        batch_size=BATCH_SIZE,
                                                                        shuffle=True)

#Loading validation set data as image generator - rescaling 1./255
validation_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(directory=val_set, 
                                                                         class_mode='binary', 
                                                                         target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                         color_mode="grayscale",
                                                                         batch_size=BATCH_SIZE,
                                                                         shuffle=True)

#Model creation; AlexNet-like CNN with 40% dropout

# def createModel():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(64, 11, activation='relu', padding='same', 
#                                                 input_shape=[IMG_WIDTH,IMG_HEIGHT, 1]),
#         tf.keras.layers.MaxPooling2D(2),
#         tf.keras.layers.Conv2D(128, 5, activation='relu', padding='valid'),
#         tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#         tf.keras.layers.MaxPooling2D(2),
#         tf.keras.layers.Conv2D(256, 3, activation='relu', padding='valid'),
#         tf.keras.layers.Conv2D(256, 3, activation='relu', padding='valid'),
#         tf.keras.layers.MaxPooling2D(2),
#         tf.keras.layers.Conv2D(512, 3, activation='relu', padding='valid'),
#         tf.keras.layers.Conv2D(512, 3, activation='relu', padding='valid'),
#         tf.keras.layers.MaxPooling2D(2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.4),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(1, activation='softmax')
#     ])

def createModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 5, activation='relu', 
                                                input_shape=[IMG_WIDTH,IMG_HEIGHT, 1]),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 5, activation='relu'),
        tf.keras.layers.Conv2D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu'),
        tf.keras.layers.Conv2D(256, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    sgd = tf.keras.optimizers.SGD(lr=1e-6)
    model.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(0.001),
                    metrics=['accuracy'])
    model.summary()

    #tf.keras.utils.plot_model(model, to_file='plots/model_structure.png', show_shapes=True)

    return model

#Model learning 

def fitModel(model,trainGen, epoch, stepsPE, validationGen, stepsVal):
    history = model.fit(trainGen, 
                        epochs=epoch,
                        steps_per_epoch = stepsPE, 
                        validation_data=validationGen,
                        validation_steps = stepsVal,
                        callbacks=tensor_board+callbacks)

    return history

fitModel(createModel(), trainGen=train_generator, epoch=EPOCHS, stepsPE=train_samples//BATCH_SIZE,
 validationGen=validation_generator, stepsVal=validation_samples//BATCH_SIZE)

#training for arrays?

#train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# fitModel(createModel(), trainGen=train_generator.flow(train_tensor, train_label, batch_size=BATCH_SIZE, subset='training'),
#  epoch=EPOCHS, stepsPE=100, validationGen=validation_generator.flow(val_tensor, val_label, batch_size=BATCH_SIZE, subset='validation'), 
#  stepsVal=15)