import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn import metrics
from sklearn.model_selection import learning_curve
import tensorflow as tf
import matplotlib.pyplot as plt

EPOCHS = 30
BATCH_SIZE = 15
IMG_WIDTH, IMG_HEIGHT = 128, 128

#train_set = 
#test_set = 

#Loading train set data as image generator - rescaling 1./255
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(directory=train_set, 
                                                                        class_mode='categorical', 
                                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                        color_mode="grayscale",
                                                                        batch_size=BATCH_SIZE
                                                                        shuffle=True)

#Loading test set data as image generator - rescaling 1./255
validation_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(directory=test_set, 
                                                                         class_mode='categorical', 
                                                                         target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                         color_mode="grayscale",
                                                                         batch_size=BATCH_SIZE
                                                                         shuffle=True)

#Model creation; AlexNet-like CNN with 40% dropout

def createModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 11, activation='relu', padding='same', 
                                                input_shape=[IMG_WIDTH,IMG_HEIGHT, 1]),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 5, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='valid'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(512, 3, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(512, 3, activation='relu', padding='valid'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(loss="binary_crossentropy",
                    optimizer=opt,
                    metrics=['accuracy'])
    model.summary()

    return model


def fitModel(model,trainGen,epoch, stepsPE, validationGen, stepsVal):
    history = model.fit(trainGen, 
                        epochs=epoch,
                        steps_per_epoch = stepsPE, 
                        validation_data=validationGen,
                        validation_steps = stepsVal)

    return history

fitModel(createModel(), trainGen=train_generator, epoch=EPOCHS, stepsPE=100, validationGen=validation_generator, stepsVal=15)

#Feature Extractor; VGG16 model optimized by Adamax and ResNet50

def createVGG16Model():
    vgg = tf.keras.applications.VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)) #3 bands ok; 1 band error

    for layers in vgg.layers:
        layers.trainable=False

    print(vgg.output)

    output = vgg.get_layer('fc2').output
    output = tf.keras.layer.Flatten(name='flatten')(output)
    output = tf.keras.layer.Dense(units=1024, activation='relu', name='new_fc')(output)
    output = tf.keras.layer.Dense(units=10, activation='softmax')(output)
    vgg = tf.keras.models.Model(vgg.input, output)

    #Compiling the vgg16 model
    opt_vgg = tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    vgg.compile(optimzer=opt_vgg, loss='binary_crossentropy', metrics=['accuracy'])
    vgg.summary()

    return vgg

def createResNet50():
    resnet = tf.keras.applications.ResNet50(weights='imagenet', 
                                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 1), #3 bands ok; 1 band error
                                            include_top=True)
    for layer in resnet.layers[:]:
        layer.trainable = False

    output = resnet.get_layer('avg_pool').output
    output = tf.keras.layer.Flatten(name='flatten')(output)
    output = tf.keras.layer.Dense(units=1024, activation='relu', name='fc')(output)
    output = tf.keras.layer.Dense(units=10, activation='softmax')(output)
    resnet = tf.keras.models.ModelModel(resnet.input, output)

    #Compiling the resnet50 model
    opt_resnet = tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    resnet.compile(optimizer=opt_resnet, loss='binary_crossentropy', metrics=['accuracy'])
    resnet.summary()
    
    return resnet
