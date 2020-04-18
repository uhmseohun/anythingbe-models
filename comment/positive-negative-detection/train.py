from generate_model import generate_model
from pre_processing import data

model = generate_model()
(x_train, y_train) = data

model.fit(
  x_train, y_train,
  epochs=30, verbose=1
)
