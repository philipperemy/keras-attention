from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

from attention_utils import get_activations, get_data_recurrent

INPUT_DIM = 2
TIME_STEPS = 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = True
APPLY_ATTENTION_BEFORE_LSTM = False


def attention_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='attention_vec')(a)  # this is the attention vector!
        a = RepeatVector(input_dim)(a)
    else:
        a = Lambda(lambda x: x, name='attention_vec')(a)  # trick to name a layer.
    a_probs = Permute((2, 1))(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)

    # ATTENTION PART STARTS HERE
    attention_mul = attention_block(lstm_out, lstm_units)
    # a = Permute((2, 1))(lstm_out)
    # a = Reshape((lstm_units, TIME_STEPS))(a)
    # a = Dense(TIME_STEPS, activation='softmax')(a)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='attention_vec')(a)  # this is the attention vector!
    #     a = RepeatVector(lstm_units)(a)
    # else:
    #     a = Lambda(lambda x: x, name='attention_vec')(a)  # trick to name a layer.
    # a_probs = Permute((2, 1))(a)
    # attention_mul = merge([lstm_out, a_probs], name='attention_mul', mode='mul')
    # ATTENTION PART FINISHES HERE

    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32

    # ATTENTION PART STARTS HERE
    attention_mul = attention_block(inputs, INPUT_DIM)
    # a = Permute((2, 1))(inputs)
    # a = Reshape((INPUT_DIM, TIME_STEPS))(a)
    # a = Dense(TIME_STEPS, activation='softmax')(a)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='attention_vec')(a)  # this is the attention vector!
    #     a = RepeatVector(INPUT_DIM)(a)
    # else:
    #     a = Lambda(lambda x: x, name='attention_vec')(a)  # trick to name a layer.
    # a_probs = Permute((2, 1))(a)
    # attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    # ATTENTION PART FINISHES HERE

    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':

    N = 300000
    # N = 300 -> too few = no training
    inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)

    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_after_lstm()
    else:
        m = model_attention_applied_before_lstm()

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)

    attention_vectors = []
    for i in range(300):
        testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        attention_vector = np.mean(get_activations(m,
                                                   testing_inputs_1,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec')[0], axis=1).squeeze()
        print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
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
