import tensorflow as tf
import numpy as np

import csv
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

with open('C:\\Users\\CPT4273\\Downloads\\NN_test\\regression_in_keras\\s.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
#  ['20221115', '47', '2', '19', '20', '26', '29', '34', '39', '4']
# print(data)

arr = np.array(data)
arr = arr.astype(np.int)
tr = arr[0:200,:]
te  = arr[200:244,:]
# # Access the First and Last column of array
# X_train = arr[:,[1,2]]
# y_train = arr[:,[3,4,5,6,7,8,9]]
# X_test = arr[:,[1,2]]
# y_test = arr[:,[3,4,5,6,7,8,9]]

X_train = tr[:,[1,2]]
y_train = tr[:,[3,4,5,6,7,8]]
X_test = te[:,[1,2]]
y_test = te[:,[3,4,5,6,7,8]]


print(X_train.shape)

print(X_test.shape)

print(X_train[0])

print(y_train[0])

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Dense(64, input_dim=2, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())

# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.2))

# model.add(tf.keras.layers.Dense(7, activation='linear'))

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(4096, input_dim=2, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Dense(6, activation='linear'))

model.summary()

Model: "sequential"

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train,
          batch_size=64, epochs=128, verbose=1,
          validation_data=(X_test, y_test))


# test = [1, 5]
# print(model.predict(np.reshape(test, [1, 2])).astype('int'))
test = [51, 5]
print(model.predict(np.reshape(test, [1, 2])).astype('int'))

1/0

target = []
target1 = []
target2 = []
target3 = []
target4 = []
for a in X_train:
 res = (model.predict(np.reshape(a, [1, 2])).astype('int'))
 x = res[:,[6]]
 target.append(x)
 x = res[:,[1]]
 target1.append(x)
 x = res[:,[2]]
 target2.append(x)

# plotting the y_test vs y_pred
Y_pred = np.array(target)
# print(Y_pred[:,[0]])
plt.scatter(Y_pred, X_train[:,[0]])
plt.xlabel('Y_pred')
plt.ylabel('Y_test')
plt.show()

# Y_pred = np.array(target1)
# print(Y_pred[:,[0]])
# plt.scatter(Y_pred, X_train[:,[0]])
# plt.xlabel('Y_pred')
# plt.ylabel('Y_test')
# plt.show()