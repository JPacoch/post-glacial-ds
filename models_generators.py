import tensorflow as tf

from callbacks import tensor_board, callbacks


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
                    optimizer=tf.keras.optimizers.Adam(0.00001),
                    metrics=['accuracy'])
    model.summary()

    tf.keras.utils.plot_model(model, to_file='plots/model_structure.png', show_shapes=True)

    return model

def fitModel(model,trainGen, epoch, stepsPE, validationGen, stepsVal):
    history = model.fit(trainGen, 
                        epochs=epoch,
                        steps_per_epoch = stepsPE, 
                        validation_data=validationGen,
                        validation_steps = stepsVal,
                        callbacks=tensor_board+callbacks)

    return history