import numpy as np
import tensorflow as tf

Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Conv2D = tf.keras.layers.Conv2D
Input = tf.keras.layers.Input
BatchNormalization = tf.keras.layers.BatchNormalization
Flatten = tf.keras.layers.Flatten
Activation = tf.keras.layers.Activation
Reshape = tf.keras.layers.Reshape
Conv2DTranspose = tf.keras.layers.Conv2DTranspose
UpSampling2D = tf.keras.layers.UpSampling2D
RMSprop = tf.keras.optimizers.RMSprop


import os
import pandas as pd
from matplotlib import pyplot as plt


data_dir = 'quickdraw_data'
input_images = data_dir + "/apple.npy"

data = np.load(input_images)

print(data.shape)

print(data[4242])

data = data / 255
data = np.reshape(data, (data.shape[0], 28, 28, 1))
img_w, img_h = data.shape[1:3]
print(data.shape)

plt.imshow(data[4242, :, :, 0], cmap='Greys')


# discriminator

def build_discriminator(depth=64, p=0.4):
    # 定義輸入
    image = Input((img_w, img_h, 1))

    # 卷積層
    conv1 = Conv2D(depth * 1, 5, strides=2,
                   padding='same', activation='relu')(image)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(depth * 2, 5, strides=2,
                   padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth * 4, 5, strides=2,
                   padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(depth * 8, 5, strides=1,
                   padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))

    # 輸出層
    prediction = Dense(1, activation='sigmoid')(conv4)

    # 定義模型
    model = Model(inputs=image, outputs=prediction)

    return model


discriminator = build_discriminator()

discriminator.summary()

Model: "model"

discriminator.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=0.0008,
                                        decay=6e-8,
                                        clipvalue=1.0),
                      metrics=['accuracy'])

# generator

z_dimensions = 32


def build_generator(latent_dim=z_dimensions,
                    depth=64, p=0.4):
    # 定義輸入
    noise = Input((latent_dim,))

    # 第 1 密集層
    dense1 = Dense(7 * 7 * depth)(noise)
    dense1 = BatchNormalization(momentum=0.9)(dense1)  # default momentum for moving average is 0.99
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7, 7, depth))(dense1)
    dense1 = Dropout(p)(dense1)

    # 反卷積層
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth / 2),
                            kernel_size=5, padding='same',
                            activation=None, )(conv1)
    conv1 = BatchNormalization(momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth / 4),
                            kernel_size=5, padding='same',
                            activation=None, )(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    conv3 = Conv2DTranspose(int(depth / 8),
                            kernel_size=5, padding='same',
                            activation=None, )(conv2)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    # 輸出層
    image = Conv2D(1, kernel_size=5, padding='same',
                   activation='sigmoid')(conv3)

    # 定義模型
    model = Model(inputs=noise, outputs=image)

    return model


generator = build_generator()

generator.summary()

Model: "model_1"

z = Input(shape=(z_dimensions,))
img = generator(z)

discriminator.trainable = False

pred = discriminator(img)

adversarial_model = Model(z, pred)

adversarial_model.compile(loss='binary_crossentropy',
                          optimizer=RMSprop(learning_rate=0.0004,
                                            decay=3e-8,
                                            clipvalue=1.0),
                          metrics=['accuracy'])


def train(train_round=1000, batch=128, z_dim=z_dimensions):
    d_metrics = []
    a_metrics = []

    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0

    for i in range(train_round):

        # 從真影像資料集中取樣：
        real_imgs = np.reshape(
            data[np.random.choice(data.shape[0],
                                  batch,
                                  replace=False)],
            (batch, 28, 28, 1))

        # 生成假影像：
        fake_imgs = generator.predict(
            np.random.uniform(-1.0, 1.0,
                              size=[batch, z_dim]))

        # 將真假影像串起來, 方便一併輸入鑑別器：
        x = np.concatenate((real_imgs, fake_imgs))

        # 標籤 y, 提供給鑑別器：
        y = np.ones([2 * batch, 1])
        y[batch:, :] = 0

        # 訓練鑑別器：
        d_metrics.append(
            discriminator.train_on_batch(x, y)
        )
        running_d_loss += d_metrics[-1][0]
        running_d_acc += d_metrics[-1][1]

        # 設定對抗式神經網路的輸入雜訊與標籤 y
        # (生成器希望鑑別器能誤判為真, 所以 y=1)：
        noise = np.random.uniform(-1.0, 1.0,
                                  size=[batch, z_dim])
        y = np.ones([batch, 1])

        # 訓練對抗式神經網路：
        a_metrics.append(
            adversarial_model.train_on_batch(noise, y)
        )
        running_a_loss += a_metrics[-1][0]
        running_a_acc += a_metrics[-1][1]

        # 定時顯示進度與生成影像：
        if (i + 1) % 100 == 0:

            print('train_round #{}'.format(i))
            log_mesg = "%d: [D loss: %f, acc: %f]" % \
                       (i, running_d_loss / i, running_d_acc / i)
            log_mesg = "%s  [A loss: %f, acc: %f]" % \
                       (log_mesg, running_a_loss / i, running_a_acc / i)
            print(log_mesg)

            noise = np.random.uniform(-1.0, 1.0,
                                      size=[16, z_dim])
            gen_imgs = generator.predict(noise)

            plt.figure(figsize=(5, 5))

            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k + 1)
                plt.imshow(gen_imgs[k, :, :, 0],
                           cmap='gray')
                plt.axis('off')

            plt.tight_layout()
            plt.show()

    return a_metrics, d_metrics


a_metrics_complete, d_metrics_complete = train()

ax = pd.DataFrame(
    {
        'Generator': [metric[0] for metric in a_metrics_complete],
        'Discriminator': [metric[0] for metric in d_metrics_complete],
    }
).plot(title='Training Loss', logy=True)
ax.set_xlabel("train_round")
ax.set_ylabel("Loss")


ax = pd.DataFrame(
    {
        'Generator': [metric[1] for metric in a_metrics_complete],
        'Discriminator': [metric[1] for metric in d_metrics_complete],
    }
).plot(title='Training Accuracy')
ax.set_xlabel("train_round")
ax.set_ylabel("Accuracy")
