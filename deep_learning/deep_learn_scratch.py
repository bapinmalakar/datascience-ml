# craete model from scratch using layers and weight
# we going to load all image file from csv file rather than different image by ImageGenerator()

import numpy as np
import pandas as pd
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout

print('Setup done')


train_file = './test_data/train.csv'
raw_data = pd.read_csv(train_file)

img_rows, img_cols = 28, 28
num_classes = 10  # indentify numbers is 1, 10 or number of different values is 1 to 10


def data_pre(raw):  # to extract label and reshape the pixel intensity data back to 28X28 grade before apply modal
    # out_y, set target
    # raw.label, target value, num_classe: different type of value
    out_y = keras.utils.to_categorical(raw.label, num_classes)
    # to_categorical return 1/2 encoded version of the target, means return 10 possible value of the target with 10 seperate binary cols
    num_images = raw.shape[0]
    # work on data intensity
    x_as_array = raw.values[:, 1:]
    # raw.values, give data as numpy array
    # raw.values[:,1:], return all data except cell 1(label)
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    # reshape into 4d array, using

    # out_x set features
    out_x = x_shaped_array/255

    return out_x, out_y


x, y = data_pre(raw_data)
print(x, y)

# prepare model
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

#Conv2D, increase the number of this layer, increase abelity of the model to fit traing data

# kernel_size, canv size
# activation, activation function
# 20, number of filter in each canv layer

model.add(Flatten())
# convert the output each of layer into 1D representation for each image

model.add(Dense(128, activation="relu"))  # 128, nodes in dense layer

# final layer with activation function and node I want finally
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

#all data loaded into arrays, so I use fit rather than fit_generator
model.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split=0.2)
#validation_split=0.2, 20% data should be satisfy the velidation and living 80% for traing data