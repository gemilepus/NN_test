import tensorflow as tf

vgg19 = tf.keras.applications.VGG19(include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3),
                                    pooling=None)

for layer in vgg19.layers:
    layer.trainable = False

model = tf.keras.models.Sequential()
model.add(vgg19)

model.add(tf.keras.layers.Flatten(name='flattened'))
model.add(tf.keras.layers.Dropout(0.7, name='dropout'))
model.add(tf.keras.layers.Dense(2, activation='softmax', name='predictions'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    data_format='channels_last',
    rotation_range=30,
    horizontal_flip=True,
    fill_mode='reflect')

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    data_format='channels_last')

batch_size = 8

train_generator = train_datagen.flow_from_directory(
    directory='data',
    target_size=(224, 224),
    classes=['y_flower', 'not_y_flower'],
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=42)

valid_generator = valid_datagen.flow_from_directory(
    directory='test',
    target_size=(224, 224),
    classes=['y_flower', 'not_y_flower'],
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=42)

model.fit(train_generator, steps_per_epoch=10,
          epochs=8, validation_data=valid_generator,
          validation_steps=10)

