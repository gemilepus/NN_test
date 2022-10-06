import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

print(X_train.shape)

print(X_test.shape)

print(X_train[0])

print(y_train[0])

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(32, input_dim=13, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(1, activation='linear'))

model.summary()

Model: "sequential"

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train,
          batch_size=8, epochs=32, verbose=1,
          validation_data=(X_test, y_test))

print(X_test[42])

print(y_test[42])

model.predict(np.reshape(X_test[42], [1, 13]))
