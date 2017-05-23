import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import *
from keras.layers.core import *
from keras.layers import Input, Dense, merge

input_dims = 32


def get_activations(model, inputs, print_shape_only=False):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    if len(inputs.shape) == 3:
        batch_inputs = inputs[np.newaxis, ...]
    else:
        batch_inputs = inputs
    layer_outputs = [func([batch_inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def build_model():
    inputs = Input(shape=(input_dims,))

    # ATTENTION PART STARTS HERE
    alpha = Dense(32, activation='softmax', name='alpha')(inputs)
    r = merge([inputs, alpha], output_shape=32, name='r', mode='mul')
    # ATTENTION PART FINISHES HERE

    r = Dense(64)(r)
    output = Dense(1, activation='sigmoid')(r)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':
    def get_data(n, attention_column=1):
        """
        Data generation. x is purely random except that it's first value equals the target y.
        In practice, the network should learn that the target = x[attention_column].
        Therefore, most of its attention should be focused on the value addressed by attention_column.
        :param n: the number of samples to retrieve.
        :return: x: model inputs, y: model targets
        """
        x = np.random.standard_normal(size=(n, input_dims))
        y = np.random.randint(low=0, high=2, size=(n, 1))
        x[:, attention_column] = y[:, 0]
        return x, y


    N = 10000
    inputs_1, outputs = get_data(N)

    m = build_model()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit([inputs_1], outputs, epochs=20, batch_size=64, validation_split=0.5)

    testing_inputs_1, testing_outputs = get_data(1)

    # Attention vector corresponds to the second matrix.
    # The first one is the Inputs output.
    attention_vector = get_activations(m, testing_inputs_1, print_shape_only=True)[1].flatten()
    print('attention =', attention_vector)

    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                                   title='Attention Mechanism as '
                                                                         'a function of input'
                                                                         ' dimensions.')
    plt.show()
