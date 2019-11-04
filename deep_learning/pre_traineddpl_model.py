# we are going to use pre trained deep learnig model first
# install virtualenv, sudo pip3 install -U virtualenv, if not present
# for check virtulaenv --version

# install tensorflow sudo pip3 install -U tensorflow
#install keras
#install pillow, if get PIL error, its a dependency for load_img of keras

from os.path import join
import numpy as np
from keras.applications.resnet50 import preprocess_input 
#preprocess_input, perform mathmathical operation on image pixel value, so all value should be 1 and -1

from keras.preprocessing.image import load_img, img_to_array
#load_img, use to load images, we are pass two params, 1. image path, 2. target_size: image size
#img_to_array, convert an image into array and te array is 3Tensor array of a image

from keras.applications import ResNet50
#Resnet50 is the model for image detection

from learntools.deep_learning.decode_predictions import decode_predictions
from IPython.display import Image, display

image_dir = './train_image/'

img_list = ['0a0c223352985ec154fd604d7ddceabd.jpg', '0a1b0b7df2918d543347050ad8b16051.jpg', '0a001d75def0b4352ebde8d07c0850ae.jpg',
            '0a1f8334a9f583cac009dc033c681e47.jpg', '0a3f1898556115d6d0931294876cd1d9.jpg', '0a70f64352edfef4c82c22015f0e3a20.jpg',
            '0a65ba3ab9b29c66e15cec76f34eca6f.jpg', '0a27d304c96918d440e79e6e9e245c3f.jpg']

img_paths = [join(image_dir, filename) for filename in img_list]

print('image path are: ', img_paths)

#conside each image is 224 X 224
image_size = 224

#function for prepare data for model
def  read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    #load each image present in image_paths array
    img_array = np.array([img_to_array(img) for img in imgs])
    #craete array of 3Tenser of each image, combine of multiple 3Tenser array is 4Dimensional array
    output = preprocess_input(img_array)
    #procees arithmathic operation on all input pixel value, and the vaule are in -1 or 1
    return output

#craete a model with pre-trained weight data
img_detect_model = ResNet50(weights='./pretrained_weight.h5')

#get test data
test_data = read_and_prep_images(img_paths)
print(test_data)

#prediction
predicts_result = img_detect_model.predict(test_data)
print('predicts_result \n', predicts_result)
#it will predict the probability with each image present in pre_trained file, so we need to find higest probality
#of each image to know actual image
#decode_predictions: decode your prediction and consider top 3 prediction and get class form class list path
most_likely_labels = decode_predictions(predicts_result, top=3, class_list_path='../input/resnet50/imagenet_class_index.json')
for i, img_path in enumerate(img_paths):
    display(Image(img_path)) #display image
    print(most_likely_labels[i])