import numpy as np
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
    
    def plot_fmap(self, fmap_shape, feature_map):
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

    def plot_fmaps(self, names_layer, model_predictions, img_per_row=18):

        for name_layer, model_layer in zip(names_layer, model_predictions):
            n_f = model_layer.shape[-1]
            size = model_layer.shape[1]

            n_cols = n_f // img_per_row
            grid_plot = np.zeros((size * n_cols, img_per_row * size))

            for col in range(n_cols):
                for row in range(img_per_row):
                    channel_img = model_layer[0,
                                                    :, :,
                                                    col * img_per_row + row]
                    channel_img -= channel_img.mean()
                    channel_img /= channel_img.std()
                    channel_img *= 64
                    channel_img += 128
                    channel_img = np.clip(channel_img, 0, 255).astype('uint8')
                    grid_plot[col * size : (col + 1) * size,
                                row * size : (row + 1) * size] = channel_img

            scale = 1. / size
            plt.figure(figsize=(scale * grid_plot.shape[1],
                                scale * grid_plot.shape[0]))
            plt.title(name_layer)
            plt.grid(False)
            plt.imshow(grid_plot, aspect='auto', cmap='viridis')
            
        plt.show()