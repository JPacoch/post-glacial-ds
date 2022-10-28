import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
import matplotlib.pyplot as plt
from turtle import color
from sklearn import metrics
from sklearn.model_selection import learning_curve

from callbacks import tensor_board, callbacks
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

#Model creation; AlexNet-like CNN with 40% dropout

def createModel(IMG_WIDTH, IMG_HEIGHT):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, 3, activation='relu', 
                                                input_shape=[IMG_WIDTH,IMG_HEIGHT,1]),
        tf.keras.layers.BatchNormalization(),                                        
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(0.000001),
                    metrics=['accuracy'])
    model.summary()

    tf.keras.utils.plot_model(model, to_file='plots/model_structure.png', show_shapes=True)

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

history = fitModel(createModel(IMG_WIDTH, IMG_HEIGHT), trainGen=train_generator, epoch=EPOCHS, stepsPE=train_samples//BATCH_SIZE,
 validationGen=test_generator, stepsVal=test_samples//BATCH_SIZE)

#training for arrays?

#train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# fitModel(createModel(), trainGen=train_generator.flow(train_tensor, train_label, batch_size=BATCH_SIZE, subset='training'),
#  epoch=EPOCHS, stepsPE=100, validationGen=validation_generator.flow(val_tensor, val_label, batch_size=BATCH_SIZE, subset='validation'), 
#  stepsVal=15)

plt.plot(history.history['accuracy'], color = 'darkblue')
plt.plot(history.history['val_accuracy'], color = 'lightblue')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()