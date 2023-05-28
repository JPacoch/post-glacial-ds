import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

from plot import PlotModel
from utils import conf_matrix
from callbacks import tensor_board, callbacks
from models_generators import createModel, fitModel
from config import EPOCHS, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, FMAP_SITES


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

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu', 
                                            input_shape=[IMG_WIDTH,IMG_HEIGHT,1]),
    tf.keras.layers.BatchNormalization(),                                        
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(0.000001),
                metrics=['accuracy'])
model.summary()

#Model training 
history = fitModel(model, trainGen=train_generator, epoch=EPOCHS, stepsPE=train_samples//BATCH_SIZE,
 validationGen=test_generator, stepsVal=test_samples//BATCH_SIZE)

model.save('models/june1.h5')

plotCls = PlotModel()
plotCls.plot_training(history=history)

y_pred = model.predict_generator(train_generator, train_generator.samples // train_generator.batch_size+1)
false_positive, true_positive, ths = roc_curve(train_generator.classes, y_pred)
auc = auc(false_positive, true_positive)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(false_positive, true_positive, 'k', label='area = {:.3f}'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

conf_matrix(train_generator, model=model)

#Feature extraction
trainpath = 'points/train/'

#imgpath = 'points/train/nondenuded/nondenuded89.png'
for path in FMAP_SITES:
    imgpath = trainpath + path + '.png'
    image = tf.keras.utils.load_img(imgpath, target_size=(128,128), color_mode='grayscale')
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    # for i in range(len(model.layers)):
    #     layer = model.layers[i]
    #     if 'conv' not in layer.name:
    #         continue    
    #     print(i , layer.name , layer.output.shape)

    # for fmapping multiple layers
    # ixs = [0, 3, 7]
    # outputs = [model.layers[i+1].output for i in ixs]
    # model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs)

    # for single layer use
    # model = tf.keras.models.Model(inputs=model.inputs , outputs=model.layers[1].output)

    # feature_maps = model.predict(image)

    # plotCls.plot_fmap(fmap_shape=FMAP_SHAPE, feature_map=feature_maps)

    layer_outputs = [layer.output for layer in model.layers[:10]]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(image)

    layer_names = []
    for layer in model.layers[:17]:
        layer_names.append(layer.name)

    plotCls.plot_fmaps(layer_names, activations, imgname=path, img_per_row=12)