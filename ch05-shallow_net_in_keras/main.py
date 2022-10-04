import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("X_train.shape:")
print(X_train.shape)

np.set_printoptions(linewidth=np.inf)

print("X_train[0]:")
print(X_train[0])
print("y_train.shape:")
print(y_train.shape)
print("y_train[0:12]:")
print(y_train[0:12])

plt.figure(figsize=(5,5))
for k in range(12):
  plt.subplot(3, 4, k+1)
  plt.imshow(X_train[k], cmap='gray')
plt.tight_layout()
plt.show()

X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')

X_train /= 255
X_test /= 255

print("X_train[0]:")
print(X_train[0])

n_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(784, )))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              metrics=['accuracy'])

model.fit(X_train,y_train,
          batch_size=128,
          epochs=200,
          verbose=1,
          validation_data=(X_test, y_test))