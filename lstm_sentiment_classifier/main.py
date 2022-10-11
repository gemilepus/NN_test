import tensorflow as tf

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
imdb = tf.keras.datasets.imdb
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
Embedding = tf.keras.layers.Embedding
SimpleRNN = tf.keras.layers.SimpleRNN
LSTM = tf.keras.layers.LSTM
SpatialDropout1D = tf.keras.layers.SpatialDropout1D
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

output_dir = 'LSTM'

epochs = 4
batch_size = 128

n_dim = 64
n_unique_words = 10000
max_review_length = 100
pad_type = trunc_type = 'pre'
drop_embed = 0.2

n_lstm = 256
drop_lstm = 0.2

# n_dense = 256
# dropout = 0.2

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n_unique_words)

x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
x_test = pad_sequences(x_test, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)

model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(SpatialDropout1D(drop_embed))
model.add(LSTM(n_lstm, dropout=drop_lstm))
# model.add(Dense(n_dense, activation='relu'))
# model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

model.summary()

Model: "sequential_1"

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modelcheckpoint = ModelCheckpoint(filepath=output_dir + "/weights.{epoch:02d}.hdf5")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
          callbacks=[modelcheckpoint])

model.load_weights(output_dir + "/weights.02.hdf5")  # 請視以上執行結果指定較佳的權重

y_hat = model.predict(x_test)

plt.hist(y_hat)
plt.axvline(x=0.5, color='orange')
plt.show()

"{:0.2f}".format(roc_auc_score(y_test, y_hat) * 100.0)
