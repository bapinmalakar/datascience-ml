#transfer learning, procees to learn from pretrained data model for new dataset or new model

# allow tensorflow to learn and make predict
# because here we dont have any pretrained data
# ResNet have n number of layer, we cut last layer, take single layer
# a single layer is collection of vetor, each point known as node
# then we put another layer which is new, not present in pretarained layer(wants to predict)
# now, try to find the reletionship between all node present in last layer and our new layer

# example if find the photo take in Urban or Rural area, which not present in pretrained data

from keras.models import Sequential
# we going to have model which is sequence of layer one after another

from keras.layers import Dense
# for add dense layer

from keras.preprocessing.image import ImageDataGenerator
# to generate subdirectory of images

from keras.applications.resnet50 import preprocess_input

from keras.applications import ResNet50

# in this example we going to classified images into two category Urban or Rural, we save them as num_classes

# pretrained ResNet50 model
resnet_weight_path = './pretrained_mode_data/for_rular_urban.h5'
num_classes = 2  # we need two classes(Urban and rural)

new_model = Sequential()
# frist we add pretrained ResNet50 model
new_model.add(ResNet50(include_top=False,
                       weights=resnet_weight_path, pooling='avg'))

# include_top=False define want to exclude last layer which make prediction
# pooling='avg', define at last we want collaps by taking avg fro 1D tenser

# add dense layer to prediction
new_model.add(Dense(num_classes, activation='softmax'))
# num_class define number of node in the new layer
# activation='softmax', apply softmax function to define probability from channels

new_model.layers[0].trainable = False
# dont trained first laye, because this layer already pretrainde ResNet50 model layers

# compile, command tell tensorflow how to update relationship of Dense connection when you train the data
new_model.compile(
    optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# metrics=['accuracy'], want accuracy not loss
# loss='categorical_crossentropy', categorical_crossentropy algorithm for minimize loss

# our row data dividen into directory
# Training Data and Validate Data
# and each we have two subdirectory Rural and Urban
# keras have a tool to divide this, know as ImageDataGenerator

# create a ImageDataGenerator

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# preprocessing_function=preprocess_input, apply ResNet preprocessing_function function everytime on Image

image_size = 224

# for training data
train_generator = data_generator.flow_from_directory(
    './test_data/urban-and-rural-photos/rural_and_urban_photos/train',
    target_size=(image_size, image_size),
    batch_size=12,
    class_mode='categorical')
# './train_image', image directory path
# target_size: image height and width
# batch_size=12, at a time 12 image
# class_mode='categorical, classified data into categorious

# for validate data
validation_generator = data_generator.flow_from_directory(
    './test_data/urban-and-rural-photos/rural_and_urban_photos/val',
    target_size=(image_size, image_size),
    batch_size=10,
    class_mode='categorical')

# fit the model
new_model.fit_generator(train_generator,
                        steps_per_epoch=3,
                        validation_data=validation_generator,
                        validation_steps=1)
#train_generator, training data from here
#steps_per_epoch=6, we have 36 images in train data in both classso, 3 steps enough to traverse all images
#validation_data=validation_generator, validate data from validation_generator
#in each class validation data are 10, so one step enough
