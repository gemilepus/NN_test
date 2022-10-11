import tensorflow as tf

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
imdb = tf.keras.datasets.imdb
Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Embedding = tf.keras.layers.Embedding
SpatialDropout1D = tf.keras.layers.SpatialDropout1D
Input = tf.keras.layers.Input
concatenate = tf.keras.layers.concatenate
Conv1D = tf.keras.layers.Conv1D
GlobalMaxPooling1D = tf.keras.layers.GlobalMaxPooling1D
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

output_dir = 'multiconv'

epochs = 4
batch_size = 128

n_dim = 64
n_unique_words = 5000
max_review_length = 400
pad_type = trunc_type = 'pre'
drop_embed = 0.2

n_conv_1 = n_conv_2 = n_conv_3 = 256
k_conv_1 = 3
k_conv_2 = 2
k_conv_3 = 4

n_dense = 256
dropout = 0.2

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n_unique_words)

x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)
x_test = pad_sequences(x_test, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)

input_layer = Input(shape=(max_review_length,),
                    dtype='int16', name='input')

# embedding:
embedding_layer = Embedding(n_unique_words, n_dim,
                            name='embedding')(input_layer)
drop_embed_layer = SpatialDropout1D(drop_embed,
                                    name='drop_embed')(embedding_layer)

# three parallel convolutional streams:
conv_1 = Conv1D(n_conv_1, k_conv_1,
                activation='relu', name='conv_1')(drop_embed_layer)
maxp_1 = GlobalMaxPooling1D(name='maxp_1')(conv_1)

conv_2 = Conv1D(n_conv_2, k_conv_2,
                activation='relu', name='conv_2')(drop_embed_layer)
maxp_2 = GlobalMaxPooling1D(name='maxp_2')(conv_2)

conv_3 = Conv1D(n_conv_3, k_conv_3,
                activation='relu', name='conv_3')(drop_embed_layer)
maxp_3 = GlobalMaxPooling1D(name='maxp_3')(conv_3)

# concatenate the activations from the three streams:
concat = concatenate([maxp_1, maxp_2, maxp_3])

# dense hidden layers:
dense_layer = Dense(n_dense,
                    activation='relu', name='dense')(concat)
drop_dense_layer = Dropout(dropout, name='drop_dense')(dense_layer)
dense_2 = Dense(int(n_dense / 4),
                activation='relu', name='dense_2')(drop_dense_layer)
dropout_2 = Dropout(dropout, name='drop_dense_2')(dense_2)

# sigmoid output layer:
predictions = Dense(1, activation='sigmoid', name='output')(dropout_2)

# create model:
model = Model(input_layer, predictions)

model.summary()

Model: "model"

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
