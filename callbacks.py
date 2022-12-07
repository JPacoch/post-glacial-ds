import tensorflow as tf

#TensorBoard callback
tensor_board = [
    tf.keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        embeddings_freq=1
    )
]

#Callbacks
callbacks = [
    tf.keras.callbacks.ProgbarLogger(count_mode='samples',
                                    stateful_metrics=None)
    # tf.keras.callbacks.EarlyStopping(patience=1,
    #                                 monitor='acc'),
    # tf.keras.callbacks.ModelCheckpoint()
]