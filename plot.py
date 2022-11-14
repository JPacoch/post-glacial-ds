import matplotlib.pyplot as plt

class PlotModel():

    def __init__(self):
        self.th = 'th'


    def plot_training(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochn = range(len(acc))

        plt.plot(epochn, acc, 'bo', label = 'Training accuracy')
        plt.plot(epochn, val_acc, 'b', label = 'Validation accuracy')
        plt.title('Validation and training accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochn, loss, 'bo', label = 'Training loss')
        plt.plot(epochn, val_loss, 'b', label = 'Validation loss')
        plt.title('Validation and training loss')
        plt.legend()

        plt.show()  

    def show_gen_img(self, traingen, img_idx, img_rng):
        for _ in range(img_rng):
            img, label = traingen.next()
            print(img.shape)   #  (1,256,256,3)
            plt.imshow(img[img_idx])
            plt.show()