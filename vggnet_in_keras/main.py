import tensorflow as tf
import tflearn.datasets.oxflower17 as oxflower17
Dense = tf.keras.layers.Dense


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.layers import BatchNormalization

# from keras.callbacks import TensorBoard

X, Y = oxflower17.load_data(one_hot=True)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten())
model.add(Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(Dense(17, activation='softmax'))

model.summary()

Model: "sequential"

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, batch_size=64, epochs=240, verbose=1, validation_split=0.1, shuffle=True)  # callbacks=[tensorbrd])