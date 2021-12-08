import random
from random import shuffle
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
import tensorflow_addons as tfa
#import tensorflow_addons.image.rotate as rotate
#import tensorflow.keras.layers.CenterCrop as tensorflow.keras.layers.CenterCrop
from PIL import Image

def getInputLabel(input_img, label_img, mode, augmentation_prob):
    """
	Diversifies training set data with rotations/cropping

	:param input_img: A PIL Image instance of a given input image.
	:param label_img: A PIL Image instance of a given label image
	:param mode: Specifies whether inputs and label being returned are for training data or testing data.
	:param augmentation_prob: Probability of diversifying the data set. Set to 0.4 for train and 0.0 for test.
	
	:return: Tuple containing:
    (Input image in tensor form [height x width x channels]),
    (Label image in tensor form [height x width x channels]),
	"""
    tensor_input = tf.convert_to_tensor(input_img)
    tensor_label = tf.convert_to_tensor(label_img)

    # Resizing input image and label while preserving aspect ratio
    aspect_ratio = tf.shape(tensor_input)[1]/tf.shape(tensor_input)[0]
    resize_range = random.randint(300,320)

    resized_input = tf.image.resize(tensor_input, [int(resize_range*aspect_ratio), resize_range])
    resized_label = tf.image.resize(tensor_label, [int(resize_range*aspect_ratio), resize_range])

    # Converting Tensor image to a PIL Image instance for subsequent image diversification
    input = tensor_to_image(resized_input)
    label = tensor_to_image(resized_label)
    
    p_transform = random.random() # Generate a random number between 0.0 and 1.0 to be compared to augmentation_prob
    if mode == 'train' and p_transform <= augmentation_prob:
            
        rotation_possibilities =[0, 90, 180, 270]
        rotation_option = random.randint(0,3)
        rotation_degree = rotation_possibilities[rotation_option]

        if rotation_degree==90 or rotation_degree==270:
            aspect_ratio = 1/aspect_ratio

        random_rot1 = random.randint(-rotation_degree, rotation_degree)
        if random_rot1 is not 0 or 360:
            input = input.rotate(random_rot1)
            label = label.rotate(random_rot1)

        """
        random_rot2 = random.randint(-10,10)
        if random_rot2 is not 0:
            print("randomrot2:", random_rot2)
            input = input.rotate(input, random_rot2)
            label = label.rotate(label, random_rot2)
        """

        """"
        random_crop = random.randint(250,270)
        crop = tf.keras.layers.CenterCrop(int(random_crop*aspect_ratio), random_crop)
        input = crop(input)
        label = crop(label)
        #input = tf.keras.layers.CenterCrop(input, int(random_crop*aspect_ratio), random_crop)
        #label = tf.keras.layers.CenterCrop(label, int(random_crop*aspect_ratio), random_crop)
        """

        shift_left = random.randint(0,20)
        shift_up = random.randint(0,20)
        shift_right = input.size[0] - random.randint(0,20)
        shift_down = input.size[1] - random.randint(0,20)
        
        input = input.crop(box=(shift_left, shift_up, shift_right, shift_down))
        label = label.crop(box=(shift_left, shift_up, shift_right, shift_down))

        input = tf.convert_to_tensor(input)
        label = tf.convert_to_tensor(label)

        if random.random() > 0.5:
            input = tf.image.flip_up_down(input)
            label = tf.image.flip_up_down(label)

        if random.random() > 0.5:
            input = tf.image.flip_left_right(input)
            label = tf.image.flip_left_right(label)
    
        input = tf.image.adjust_contrast(input, 0.2)
        label = tf.image.adjust_contrast(label, 0.2)

    input = tf.image.resize(input, [int(256*aspect_ratio)-int(256*aspect_ratio)%16, 256])
    label = tf.image.resize(label, [int(256*aspect_ratio)-int(256*aspect_ratio)%16, 256])

    # Add blur and noise for further image augmentation 

    input = tf.image.convert_image_dtype(input, dtype=tf.float32)
    label = tf.image.convert_image_dtype(label, dtype=tf.float32)

    input = tf.image.per_image_standardization(input)

    return input, label

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    return Image.fromarray(tensor)

def get_data(input_path, label_path, num_inputs, image_size=224, mode='train', augmentation_prob=0.4):
    """
    :param input_img: File path for input images.
	:param label_img: File path for label images.
	:param mode: Specifies whether inputs and label being returned are for training data or testing data.
	:param augmentation_prob: Probability of diversifying the data set. Set to 0.4 for train and 0.0 for test.

    :return: Tuple containing:
    (2-d list of input images in tensor form [height x width x channels]),
	(2-d list of label images in tensor form [height x width x channels]),
    """
    inputs = []
    labels = []
    for file_name in os.listdir(input_path):
        input_img = tf.keras.preprocessing.image.load_img(input_path + '/' + file_name)
        label_img = tf.keras.preprocessing.image.load_img(label_path + '/' + file_name[:-len(".jpg")] + '_Segmentation.png')
        processed_input, processed_label = getInputLabel(input_img, label_img, mode=mode, augmentation_prob=augmentation_prob)
        inputs.append(processed_input)
        labels.append(processed_label)
    return inputs, labels
