# train our model fro both original iamge and mirror imagae

from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

pretrained_weight = './pretrained_mode_data/for_rular_urban.h5'
num_classes = 2

new_model = Sequential()

# add resnet50 model with pretrained resnet50 model and ignore last layer
new_model.add(ResNet50(weights=pretrained_weight,
                       include_top=False, pooling='avg'))

# add dense layer, means new layer for new classes
new_model.add(Dense(num_classes, activation='softmax'))

# ignore to train resnet50 layer means first layer
new_model.layers[0].trainable = False

# compile the model and set accuracy
new_model.compile(
    optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

imageSize = 224

data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                             horizontal_flip=True,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2)
# horizontal_flip, randomly decide image need flip or not befor send image to model for train
# width_shift_range and height_shift_range for crop the image
train_data = data_generator_with_aug.flow_from_directory('./test_data/urban-and-rural-photos/rural_and_urban_photos/train',
                                                         target_size=(
                                                             imageSize, imageSize),
                                                         batch_size=12,
                                                         class_mode='categorical')

# for validate data we dont need mirror image check
data_generator_without_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input)
validate_data = data_generator_without_aug.flow_from_directory('./test_data/urban-and-rural-photos/rural_and_urban_photos/val',
                                                               target_size=(
                                                                   imageSize, imageSize),
                                                               batch_size=10,
                                                               class_mode='categorical')

new_model.fit_generator(train_data,
                        steps_per_epoch=3,
                        epochs=2,
                        validation_data=validate_data,
                        validation_steps=1)

#epochs=2, means its go through each image file 2 times
#now you can same file process two time
