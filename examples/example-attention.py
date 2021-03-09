import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model, Model

from attention import Attention


def main():
    # Dummy data. There is nothing to learn in this example.
    num_samples, time_steps, input_dim, output_dim = 100, 10, 1, 1
    data_x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    data_y = np.random.uniform(size=(num_samples, output_dim))

    # Define/compile the model.
    model_input = Input(shape=(time_steps, input_dim))
    x = LSTM(64, return_sequences=True)(model_input)
    x = Attention(32)(x)
    x = Dense(1)(x)
    model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())

    # train.
    model.fit(data_x, data_y, epochs=10)

    # test save/reload model.
    pred1 = model.predict(data_x)
    model.save('test_model.h5')
    model_h5 = load_model('test_model.h5')
    pred2 = model_h5.predict(data_x)
    np.testing.assert_almost_equal(pred1, pred2)
    print('Success.')


if __name__ == '__main__':
    main()
