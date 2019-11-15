# example of Stride Lengths and Dropout
# Stride Lengths: Technique to speedup our model and reduce memory consumption
# Dropout: Technique to comeback from overfitting. 
# Its randomly take node for pre part of traning and ignore node randomly for the next part of training
#this process allow all node to prediction rather than one node dominate the prediction

#this are very useful for large model

# allow to process multiple columne and row at a time so reduce our processing time

import keras as keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout

train_file = './test_data/train.csv'
raw_data = pd.read_csv(train_file)

img_rows, img_cols = 28, 28
num_classes = 10


def data_pre(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)
    num_images = raw.shape[0]
    x_as_array = raw.values[:, 1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array/255

    return out_x, out_y


x, y = data_pre(raw_data)

model = Sequential()

model.add(Conv2D(20,
                 kernel_size=(3, 3),
                 strides=2,
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

model.add(Dropout(0.5))
#Dropout(0.5), each Conv in the preceding layer should be disconnect from subsequent layer, 50% time during model training 
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dense(num_classes, activation="softmax"))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split=0.2)
