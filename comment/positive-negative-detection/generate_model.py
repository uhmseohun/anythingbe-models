from keras.models import Sequential
from keras.layers\
  import Embedding, Dense, Conv1D, MaxPool1D, Dropout, Flatten
from keras.optimizers import Adam

input_dim = 1000
output_dim = 100
filter_count = 80
filter_size = 3

def generate_model():
  model = Sequential()
  # Temporary constant value
  model.add(Embedding(4924, output_dim, input_length=37))
  model.add(Dropout(0.15))

  model.add(Conv1D(
    filter_count,
    filter_size,
    padding='valid',
    activation='relu',
    strides=1
  ))
  model.add(Conv1D(
    filter_count,
    filter_size,
    padding='valid',
    activation='relu',
    strides=1
  ))
  model.add(MaxPool1D())

  model.add(Dense(80, activation='relu'))
  model.add(Flatten())
  model.add(Dense(1, activation='softmax'))

  model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
  )

  model.summary()

  return model
