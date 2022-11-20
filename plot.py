import matplotlib.pyplot as plt

class PlotModel():

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
            print(img.shape)
            plt.imshow(img[img_idx])
            plt.show()
    
    def plot_fmaps(self, fmap_shape, feature_map):
        for fmap in feature_map:
            ix = 1 
            for _ in range(fmap_shape):
                for _ in range(fmap_shape):
                    ax = plt.subplot(fmap_shape, fmap_shape, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
                    ix += 1
            # plt.title(f'Feature map for Conv2D layer', loc='left')
            plt.show()
