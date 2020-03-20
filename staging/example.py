import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keract import get_activations
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from keras_attention.attention import attention_3d_block

INPUT_DIM = 100
TIME_STEPS = 20


def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.randint(input_dim, size=(n, time_steps))
    x = np.eye(input_dim)[x]
    y = x[:, attention_column, :]
    return x, y


def get_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    rnn_out = LSTM(32, return_sequences=True)(inputs)
    attention_output = attention_3d_block(rnn_out)
    output = Dense(INPUT_DIM, activation='sigmoid', name='output')(attention_output)
    m = Model(inputs=[inputs], outputs=[output])
    print(m.summary())
    return m


def main():
    n = 300000
    inputs, outputs = get_data_recurrent(n, TIME_STEPS, INPUT_DIM)

    m = get_model()
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    m.fit(x=[inputs], y=outputs, epochs=2, batch_size=64, validation_split=0)

    num_simulations = 10
    attention_vectors = np.zeros(shape=(num_simulations, TIME_STEPS))
    for i in range(num_simulations):
        testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        activations = get_activations(m, testing_inputs_1, layer_name='attention_weight')
        attention_vec = np.mean(activations['attention_weight'], axis=0).squeeze()
        assert np.abs(np.sum(attention_vec) - 1.0) < 1e-5
        attention_vectors[i] = attention_vec

    attention_vector_final = np.mean(attention_vectors, axis=0)
    attention_df = pd.DataFrame(attention_vector_final, columns=['attention (%)'])
    attention_df.plot(kind='bar', title='Attention Mechanism as a function of input dimensions.')
    plt.show()


if __name__ == '__main__':
    main()
