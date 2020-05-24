from model import text_cnn
from dataset import load_train_data, load_test_data
import numpy as np

model = text_cnn()

x_train, y_train = load_train_data()

y_train = [int(y) for y in y_train]

model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1)
