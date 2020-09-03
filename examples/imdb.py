import numpy
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing import sequence

from attention import Attention


def train_and_evaluate_model_on_imdb(add_attention=True):
    numpy.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
    # truncate and pad input sequences
    max_review_length = 500
    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
    # create the model
    embedding_vector_length = 32

    model = Sequential([
        Embedding(top_words, embedding_vector_length, input_length=max_review_length),
        Dropout(0.5),
        # attention vs no attention. same number of parameters so fair comparison.
        *([LSTM(100, return_sequences=True), Attention()] if add_attention
          else [LSTM(100), Dense(350, activation='relu')]),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ]
    )

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    class RecordBestTestAccuracy(Callback):

        def __init__(self):
            super().__init__()
            self.val_accuracies = []
            self.val_losses = []

        def on_epoch_end(self, epoch, logs=None):
            self.val_accuracies.append(logs['val_accuracy'])
            self.val_losses.append(logs['val_loss'])

    rbta = RecordBestTestAccuracy()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, callbacks=[rbta])

    print(f"Max Test Accuracy: {100 * np.max(rbta.val_accuracies):.2f} %")
    print(f"Mean Test Accuracy: {100 * np.mean(rbta.val_accuracies):.2f} %")


def main():
    # 10 epochs.
    # Max Test Accuracy: 88.02 %
    # Mean Test Accuracy: 87.26 %
    train_and_evaluate_model_on_imdb(add_attention=False)
    # 10 epochs.
    # Max Test Accuracy: 88.74 %
    # Mean Test Accuracy: 88.00 %
    train_and_evaluate_model_on_imdb(add_attention=True)


if __name__ == '__main__':
    main()
