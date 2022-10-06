import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

n_input = 784
n_dense = 256

b_init = tf.keras.initializers.Zeros()

# w_init = tf.keras.initializers.RandomNormal(stddev=1.0)
# w_init = tf.keras.initializers.glorot_normal()
w_init = tf.keras.initializers.glorot_uniform()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(n_dense,
                                input_dim=n_input,
                                kernel_initializer=w_init,
                                bias_initializer=b_init))
model.add(tf.keras.layers.Activation('sigmoid'))
# model.add(Activation('tanh'))
# model.add(Activation('relu'))


x = np.random.random((1, n_input))

a = model.predict(x)

plt.hist(np.transpose(a))
plt.show()
