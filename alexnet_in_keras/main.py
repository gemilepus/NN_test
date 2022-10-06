import tensorflow as tf
import tflearn.datasets.oxflower17 as oxflower17
Dense = tf.keras.layers.Dense
X, Y = oxflower17.load_data(one_hot=True)

print(Y[0])

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(256, kernel_size=(5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096, activation='tanh'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4096, activation='tanh'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(17, activation='softmax'))

model.summary()

Model: "sequential"

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, batch_size=64, epochs=100, verbose=1, validation_split=0.1, shuffle=True)