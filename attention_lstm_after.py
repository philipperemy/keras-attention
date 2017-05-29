from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

from attention_utils import get_activations, get_data_recurrent

input_dim = 2
time_steps = 20


def build_recurrent_model(single_attention_vector=True):
    inputs = Input(shape=(time_steps, input_dim,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)

    # ATTENTION PART STARTS HERE
    a = Permute((2, 1))(lstm_out)
    a = Reshape((lstm_units, time_steps))(a)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)  # this is the attention vector!
        a = RepeatVector(lstm_units)(a)
    a_probs = Permute((2, 1))(a)
    attention_mul = merge([lstm_out, a_probs], name='attention_mul', mode='mul')
    # ATTENTION PART FINISHES HERE

    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':

    N = 300000
    inputs_1, outputs = get_data_recurrent(N, time_steps, input_dim)

    m = build_recurrent_model()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)

    attention_vectors = []
    for i in range(300):
        testing_inputs_1, testing_outputs = get_data_recurrent(1, time_steps, input_dim)
        attention_vector = np.mean(get_activations(m, testing_inputs_1, print_shape_only=True)[4], axis=1).flatten()
        print('attention =', attention_vector)
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()
