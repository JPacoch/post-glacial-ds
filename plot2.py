import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from config import EPOCHS, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE


# print summary
model = tf.keras.models.load_model('models/new.h5')

tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


# print ReLU
def ReLU(x):
    return np.maximum(0, x)


def dReLU(x):
    return (x > 0)*1.0


x = np.linspace(-2, 2, 100)  # generate 100 points from -2 to 2
y = ReLU(x)
dy = dReLU(x)

ytick_positions = [0, 0.5, 1, 1.5, 2]
xtick_positions = [-2, -1, 0, 1, 2]

plt.yticks(ytick_positions)
plt.xticks(xtick_positions)
plt.plot(x, y, label='f (x)', color='grey')
plt.plot(x, dy, '--', label='f \'(x)', color='black')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, linewidth=0.5, color='gray', linestyle='--')
# plt.title('Funkcja aktywacji ReLU oraz przebieg jej pochodnej')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-5, 5, 100)
y_sigmoid = sigmoid(x)
y_derivative = sigmoid_derivative(x)

ytick_positions = [0, 0.5, 1, 1.5, 2]
xtick_positions = [-5, -2.5, 0, 2.5, 5]

plt.yticks(ytick_positions)
plt.xticks(xtick_positions)
plt.plot(x, y_sigmoid, label='f (x)', color='grey')
plt.plot(x, y_derivative, '--', label='f \'(x)', color='black')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, linewidth=0.5, color='gray', linestyle='--')
plt.legend()

plt.show()