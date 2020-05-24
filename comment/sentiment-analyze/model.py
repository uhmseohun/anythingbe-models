from tensorflow.keras.models import Sequential
from tensorflow.keras.layers\
    import Embedding, Dropout, Conv1D, MaxPool1D, Dense
from utils import load_vocab

vocab_size = len(load_vocab())

def text_cnn():
    model = Sequential()

    model.add(Embedding(vocab_size, 128, input_length=60))
    model.add(Dropout(0.3))

    model.add(Conv1D(256, 3, padding='valid', activation='relu', strides=1))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv1D(128, 3, padding='valid', activation='relu', strides=1))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return model
