import matplotlib.pyplot as plt

def plot_training(history):
    plt.plot(history.history['accuracy'], color = 'darkblue')
    plt.plot(history.history['val_accuracy'], color = 'lightblue')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()