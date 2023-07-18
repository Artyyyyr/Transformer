import numpy as np
import tensorflow as tf
from data import process_data_step_prediction, get_points, additional

from Transformer import Transformer

import matplotlib.pyplot as plt


def _slice_y(transformer_data, target):
    sliced_y = []
    sliced_transformer_data = []
    for i in range(1, len(target[0])):
        sliced_y.append(target[0][i - 1:i])
        sliced_transformer_data.append([transformer_data, target[:, 0:i]])
    return sliced_transformer_data, sliced_y


# Load data from path 'data/train.txt' process it
data_train = np.loadtxt('data/train.txt')
x, y = process_data_step_prediction(data_train)

# Load or create Transformer model
# transformer = Transformer.load("small_model_dense_200_encode_6")
transformer = Transformer(num_encoder=6, num_decoder=6,
                          hidden_layers_width=20, d_model=50, num_attention=5)

# print summary
transformer.summary()

# slice data for training
data = []
target = []

for i in range(len(x)):
    _data, _target = _slice_y(x[i], y[i])
    data.append(_data)
    target.append(_target)


# example of training the model
losses = transformer.train(data=data, target=target, bpoints=False, epoch=1, shuffle=True, lr=0.0001,
                           save_path="test")
np.save("model_losses_save", losses)


# example of making a sequence (generating)
id = 2
encoder_data = data[id][1]

plt.figure("Predicted")
start = [[0.]]
start = transformer.make_sequence(encoder_data, start, len(data[id]) + 30)

plt.plot(start[0])
plt.plot(target[id])

RUL = len(target[id]) - additional + 1
plt.plot([RUL, RUL],
         [0, 1])

plt.show()


# example of evaluating the model
errors = transformer.eval(data, target)
np.save("errors", errors)

plt.hist(errors)
plt.show()
