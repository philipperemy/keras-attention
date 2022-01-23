import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keract import get_activations
from keras import Input, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, LSTM

# In this example we need it because we want to extract all the intermediate output values.
os.environ['KERAS_ATTENTION_DEBUG'] = '1'
from attention import Attention

matplotlib.rcParams.update({'font.size': 8})


class VisualizeAttentionMap(Callback):

    def __init__(self, model, x):
        super().__init__()
        self.model = model
        self.x = x

    def on_epoch_begin(self, epoch, logs=None):
        attention_map = get_activations(self.model, self.x, layer_names='attention_weight')['attention_weight']
        x = self.x[..., 0]
        plt.close()
        fig, axes = plt.subplots(nrows=3, figsize=(10, 8))
        maps = [attention_map, create_argmax_mask(attention_map), create_argmax_mask(x)]
        maps_names = ['attention layer (continuous)', 'attention layer - argmax (discrete)', 'ground truth (discrete)']
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(maps[i], interpolation='none', cmap='jet')
            ax.set_ylabel(maps_names[i] + '\n#sample axis')
            ax.set_xlabel('sequence axis')
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        cbar_ax = fig.add_axes([0.75, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.suptitle(f'Epoch {epoch} - training\nEach plot shows a 2-D matrix x-axis: sequence length * y-axis: '
                     f'batch/sample axis. \nThe first matrix contains the attention weights (softmax).'
                     f'\nWe manually apply argmax on the attention weights to see which time step ID has '
                     f'the strongest weight. \nFinally, the last matrix displays the ground truth. The task '
                     f'is solved when the second and third matrix match.')
        plt.draw()
        plt.pause(0.001)


def create_argmax_mask(x):
    mask = np.zeros_like(x)
    for i, m in enumerate(x.argmax(axis=1)):
        mask[i, m] = 1
    return mask


def main():
    seq_length = 10
    num_samples = 100000
    # https://stats.stackexchange.com/questions/485784/which-distribution-has-its-maximum-uniformly-distributed
    # Choose beta(1/N,1) to have max(X_1,...,X_n) ~ U(0, 1) => minimizes amount of knowledge.
    # If all the max(s) are concentrated around 1, then it makes the task easy for the model.
    x_data = np.random.beta(a=1 / seq_length, b=1, size=(num_samples, seq_length, 1))
    y_data = np.max(x_data, axis=1)

    # NOTE: can't use Sequential in debug with KERAS_ATTENTION_DEBUG=1.
    # Because the layer Attention is no longer considered as a Keras layer.
    # That's the trick to "see" the internal outputs of each tensor in the attention module.
    # In practice, you can use Sequential without debug enabled ;)
    # model = Sequential([
    #     LSTM(128, input_shape=(seq_length, 1), return_sequences=True),
    #     Attention(),
    #     Dense(1, activation='linear')
    # ])
    model_input = Input(shape=(seq_length, 1))
    x = LSTM(128, return_sequences=True)(model_input)
    x = Attention()(x)
    x = Dense(1, activation='linear')(x)
    model = Model(model_input, x)

    model.compile(loss='mae')
    max_epoch = 100
    # visualize the attention on the first samples.
    visualize = VisualizeAttentionMap(model, x_data[0:12])
    model.fit(x_data, y_data, epochs=max_epoch, validation_split=0.2, callbacks=[visualize])


if __name__ == '__main__':
    main()
