import tensorflow as tf
import numpy as np
import pickle
import os
from data import get_points, additional


class Attention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self._name = "AttentionBlock"
        self.Dq = tf.keras.layers.Dense(d_model)
        self.Dk = tf.keras.layers.Dense(d_model)
        self.Dv = tf.keras.layers.Dense(d_model)

        self.softmax = tf.keras.layers.Softmax()
        self.d_model = d_model

    def call(self, x, q, k, v):
        if q is not None and k is not None and v is not None:
            return self._self_attention(self.Dq(q), self.Dk(k), self.Dv(v))
        else:
            return self._self_attention(self.Dv(x), self.Dv(x), self.Dv(x))

    def _self_attention(self, query, key, value):
        key = tf.transpose(key, perm=[0, 2, 1])
        mul = tf.matmul(query, key)
        w = self.softmax(mul / np.sqrt(self.d_model))

        return tf.matmul(w, value)


class EncodeBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_attention):
        super(EncodeBlock, self).__init__()
        self.num_attention = num_attention
        self.in_d_model = d_model // self.num_attention
        self.d_model = d_model
        self.attention = [Attention(self.d_model // self.num_attention) for _ in range(self.num_attention)]
        self.reshape1 = tf.keras.layers.Reshape((-1, self.num_attention, self.in_d_model))
        self.reshape2 = tf.keras.layers.Reshape((-1, self.d_model))

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()

        self.fc = tf.keras.Sequential([
                        tf.keras.layers.Dense(4 * self.d_model, activation='relu'),
                        tf.keras.layers.Dense(self.d_model)
                   ])

    def call(self, x):
        skip = x
        x = self.layernorm1(x)

        x = self._reshape(x)
        x = self._multi_attention(x)
        x = self._reshape_back(x)

        x = x + skip

        skip = x
        x = self.layernorm2(x)

        x = self.fc(x)
        return x + skip

    def _reshape(self, x):
        x = self.reshape1(x)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def _reshape_back(self, x):
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        x = self.reshape2(x)

        return x

    def _multi_attention(self, x):
        res = []
        for i, attention_layer in enumerate(self.attention):
            a = tf.gather(x, i, axis=1)
            a = attention_layer(a, None, None, None)
            res.append(a)

        return tf.stack(res)


class DecodeBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_attention):
        super(DecodeBlock, self).__init__()
        self.num_attention = num_attention
        self.in_d_model = d_model // self.num_attention
        self.d_model = d_model
        self.attention1 = [Attention(self.d_model // self.num_attention) for _ in range(self.num_attention)]
        self.attention2 = [Attention(self.d_model // self.num_attention) for _ in range(self.num_attention)]
        self.reshape1 = tf.keras.layers.Reshape((-1, self.num_attention, self.in_d_model))
        self.reshape2 = tf.keras.layers.Reshape((-1, self.d_model))

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()

        self.fc = tf.keras.Sequential([
                        tf.keras.layers.Dense(4 * self.d_model, activation='relu'),
                        tf.keras.layers.Dense(self.d_model)
                   ])

    def call(self, x, encoder_x):
        skip = x
        x = self.layernorm1(x)

        x = self._reshape(x)
        x = self._multi_attention(self.attention1, x, mask=True)
        x = self._reshape_back(x)

        x = x + skip

        skip = x
        x = self.layernorm2(x)

        x = self._reshape(x)
        encoder_x = self._reshape(encoder_x)
        x = self._multi_attention(self.attention2, x, key=encoder_x, query=x, value=encoder_x)
        x = self._reshape_back(x)

        x = x + skip

        skip = x
        x = self.layernorm2(x)

        x = self.fc(x)

        return x + skip

    def _reshape(self, x):
        x = self.reshape1(x)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def _reshape_back(self, x):
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        x = self.reshape2(x)

        return x

    def _multi_attention(self, attention, x, key=None, query=None, value=None, mask=False):
        res = []
        for i, attention_layer in enumerate(attention):
            a = tf.gather(x, i, axis=1)
            if key is not None and query is not None and value is not None:
                a = attention_layer(a,
                                    k=tf.gather(key, i, axis=1),
                                    q=tf.gather(query, i, axis=1),
                                    v=tf.gather(value, i, axis=1))

            else:
                a = attention_layer(a, None, None, None)
            res.append(a)

        return tf.stack(res)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.reshape = tf.keras.layers.Reshape((-1, self.max_sequence_length, self.d_model))

    def call(self, x):
        i = np.arange(0, self.d_model, 2)
        denominator = tf.math.pow(10000., i / self.d_model)
        position = np.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = tf.math.sin(position / denominator)
        odd_PE = tf.math.cos(position / denominator)

        even_PE = tf.reshape(even_PE, shape=(-1, 1))
        odd_PE = tf.reshape(odd_PE, shape=(-1, 1))

        stacked = tf.concat([even_PE, odd_PE], axis=1)
        stacked = tf.reshape(stacked, shape=(-1, self.d_model))
        stacked = tf.cast(stacked, dtype=tf.float32)

        shape = tf.shape(x)
        sequence_len = shape[1]

        x = x + stacked[:sequence_len]
        return x


class Transformer:
    def __init__(self, num_encoder, num_decoder, hidden_layers_width, d_model, num_attention):
        super(Transformer, self).__init__()
        self.num_encoder = num_encoder
        self.num_decoder = num_decoder
        self.hidden_layers_width = hidden_layers_width
        self.d_model = d_model
        self.num_attention = num_attention
        self._build()

    def _build(self):
        self.encoder_input = tf.keras.Input((None, 24))

        self.encoder_dense_1 = tf.keras.layers.Dense(self.hidden_layers_width, activation='relu', name="encoder_dense_1")(self.encoder_input)
        self.encoder_dense_2 = tf.keras.layers.Dense(self.hidden_layers_width, activation='relu', name="encoder_dense_2")(self.encoder_dense_1)
        self.encoder_dense_3 = tf.keras.layers.Dense(self.d_model, name="encoder_dense_3")(self.encoder_input)
        self.encoder_positional_encoding = PositionalEncoding(self.d_model, 500)(self.encoder_dense_3)

        self.encoders = [EncodeBlock(self.d_model, self.num_attention) for _ in range(self.num_encoder)]
        
        for i in range(self.num_encoder):
            self.encoder_positional_encoding = self.encoders[i](self.encoder_positional_encoding)

        self.encoder_output = self.encoder_positional_encoding

        self.decoder_input = tf.keras.Input((None, 1))
        self.decoder_dense_1 = tf.keras.layers.Dense(self.hidden_layers_width, activation='relu', name="decoder_dense_1")(self.decoder_input)
        self.decoder_dense_2 = tf.keras.layers.Dense(self.hidden_layers_width, activation='relu', name="decoder_dense_2")(self.decoder_dense_1)
        self.decoder_dense_3 = tf.keras.layers.Dense(self.d_model, name="decoder_dense_3")(self.decoder_dense_2)
        self.decoder_positional_encoding = PositionalEncoding(self.d_model, 500)(self.decoder_dense_3)

        self.decoders = [DecodeBlock(self.d_model, self.num_attention) for _ in range(self.num_decoder)]
        
        for i in range(self.num_decoder):
            self.decoder_positional_encoding = self.decoders[i](self.decoder_positional_encoding, self.encoder_output)

        self.decoder_dense_4 = tf.keras.layers.Dense(self.hidden_layers_width, activation='relu', name="decoder_dense_4")(self.decoder_positional_encoding)
        self.decoder_dense_5 = tf.keras.layers.Dense(self.hidden_layers_width, activation='relu', name="decoder_dense_5")(self.decoder_dense_4)
        self.decoder_dense_6 = tf.keras.layers.Dense(self.hidden_layers_width, name="decoder_dense_6")(self.decoder_dense_5)
        self.global_avg_pooling = tf.keras.layers.GlobalAvgPool1D()(self.decoder_dense_6)
        self.decoder_dense_7 = tf.keras.layers.Dense(self.hidden_layers_width, activation='relu', name="decoder_dense_7")(self.global_avg_pooling)
        self.decoder_dense_8 = tf.keras.layers.Dense(self.hidden_layers_width, activation='relu', name="decoder_dense_8")(self.decoder_dense_7)
        self.decoder_output = tf.keras.layers.Dense(1, activation='sigmoid', name="sigmoid")(self.decoder_dense_8)

        self.transformer = tf.keras.Model([self.encoder_input, self.decoder_input], self.decoder_output)

    def summary(self):
        self.transformer.summary()

    def train_one_engine(self, data, target, points=None, shuffle=False):
        if points is None:
            points = np.arange(0, len(data))
        if shuffle:
            np.random.shuffle(points)

        losses = []

        engine = []
        for i in points:
            history = self.transformer.fit(data[i], target[i], epochs=1)
            engine.append(*history.history['loss'])
        losses.append(engine)

        return losses

    def train(self, data, target, bpoints=False, epoch=1, shuffle=False, lr=0.0001, save_path="model"):
        op = tf.keras.optimizers.Adam(learning_rate=lr)
        loss = tf.keras.losses.BinaryCrossentropy()
        self.transformer.compile(optimizer=op, loss=loss)

        losses = []
        points = None
        print("Num of sequences: " + str(len(data)))
        for e in range(epoch):
            print("::: Epoch " + str(e + 1) + " :::")
            for i in range(len(data)):

                if bpoints:
                    points = get_points(data[i][0])
                # print(data[0][1][0])
                # print(target[0])

                print("\tSequence: " + str(i + 1))
                print("\tLength of sequence: " + str(len(data[i])))

                engine_losses = self.train_one_engine(data[i], target[i], points=points, shuffle=shuffle)

                losses.append(np.sum(engine_losses) / len(data[i]))

                self.save(save_path)

        return losses

    def eval(self, data, target):
        errors = []
        print("Length of data: " + str(len(data)))
        for i in range(len(data)):
            print("Data: " + str(i + 1))
            RUL = len(target[i]) - additional

            start = [[0.]]
            start = self.make_sequence(data[i][1], start, len(target[i]))

            for j in range(len(start[0])):
                if start[0][j] > 0.8:
                    predicted_RUL = j
                    break
            else:
                predicted_RUL = len(start[0]) + 10

            errors.append(predicted_RUL - RUL)
            print("Error: " + str(errors[-1]))

        return errors

    def _predict_next(self, encoder_input, decoder_input):
        t = self.transformer([encoder_input, np.array(decoder_input)])
        decoder_input[0].append(t[0][0].numpy())
        return decoder_input

    def make_sequence(self, data, start, sequence_len):
        sequence_len -= len(start[0])
        _start = [start[0].copy()]
        for i in range(sequence_len):
            # print(i)
            _start = self._predict_next(data[0], _start)
        return _start

    def save(self, path: os.path = "."):
        self._create_folder(path)
        self._save_parameters(path)
        self._save_weights(path)

    def _create_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _save_parameters(self, path):
        parameters = [
            self.num_encoder,
            self.num_decoder,
            self.hidden_layers_width,
            self.d_model,
            self.num_attention
        ]
        save_path = os.path.join(path, "parameters.pkl")

        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, path):
        save_path = os.path.join(path, "weights.h5")
        self.transformer.save_weights(save_path)

    def load_weights(self, weights_path):
        self.transformer.load_weights(weights_path)

    @classmethod
    def load(cls, path: os.path = "."):
        parameters_path = os.path.join(path, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        vae = Transformer(*parameters)
        weights_path = os.path.join(path, "weights.h5")
        vae.load_weights(weights_path)
        return vae


def _slice_y(transformer_data, target):
    sliced_y = []
    sliced_transformer_data = []
    for i in range(1, len(target[0])):
        sliced_y.append(target[0][i-1:i])
        sliced_transformer_data.append([transformer_data, target[:, 0:i]])
    return sliced_transformer_data, sliced_y