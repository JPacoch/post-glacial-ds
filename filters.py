import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import EPOCHS, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

model = tf.keras.models.load_model('models/test.h5')

layer_name = "conv2d_5"
layer = model.get_layer(name=layer_name)
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

def loss_compute(input_img, filter_idx):
    activ = feature_extractor(input_img)
    filter_activ = activ[:, 2:-2, 2:-2, filter_idx]
    return tf.reduce_mean(filter_activ)

@tf.function
def gradient_step(img, filter_idx, lr):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = loss_compute(img, filter_idx)
    gradients = tape.gradient(loss, img)
    gradients = tf.math.l2_normalize(gradients)
    img += lr * gradients
    return loss, img

def img_init():
    img = tf.random.uniform((1, 128, 128, 1))
    return (img - 0.5) * 0.25

def filter_vis(filter_idx):
    iter = 30
    lr = 10.0
    img = img_init()
    for iter in range(iter):
        loss, img = gradient_step(img, filter_idx, lr)
    img = img_deprocessing(img[0].numpy())
    return loss, img


def img_deprocessing(img):
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15
    img = img[25:-25, 25:-25, :]
    img += 0.5
    img = np.clip(img, 0, 1)
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

loss, img = filter_vis(0)
tf.keras.preprocessing.image.save_img("plots/singular_filter.png", img)

multiple_filters = []
for filter_idx in range(16):
    print("Processing filter %d" % (filter_idx,))
    loss, img = filter_vis(filter_idx)
    multiple_filters.append(img)

n = 4
margin = 5
crop_w = 128 - 25 * 2
crop_h = 128 - 25 * 2
width = n * crop_w + (n - 1) * margin
height = n * crop_h + (n - 1) * margin
filters_merge = np.zeros((width, height, 3))

for i in range(n):
    for j in range(n):
        img = multiple_filters[i * n + j]
        filters_merge[
            (crop_w + margin) * i : (crop_w + margin) * i + crop_w,
            (crop_h + margin) * j : (crop_h + margin) * j
            + crop_h,
            :,
        ] = img


if __name__ == '__main__':
    tf.keras.preprocessing.image.save_img("plots/filters_merged.png", filters_merge)