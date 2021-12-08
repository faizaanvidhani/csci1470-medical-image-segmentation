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

"""
class ImageFolder(tf.keras.utils.Sequence):
    
    def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.4):
        self.root = root
        self.label_paths = root[:-1]+'_GT/'
        self.input_paths = list(map(lambda x: os.path.join(root,x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.rotation = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        self.batch_size = 50
"""


def getInputLabel(input_img, label_img, mode, augmentation_prob):
    """
    Returns: Input and Label
    """
    
    #input_path = self.image_paths[index]
    #filename = input_path.split('_')[-1][:-len(".jpg")]
    #label_path = self.label_paths + 'ISIC_' + filename + '_Segmentation.png'

    #expanded_label = np.expand_dims(label_img, axis=2)
    tensor_input = tf.convert_to_tensor(input_img)
    tensor_label = tf.convert_to_tensor(label_img)

    aspect_ratio = tf.shape(tensor_input)[1]/tf.shape(tensor_input)[0]
    resize_range = random.randint(300,320)

    #print('Input Shape:', input.size)
    resized_input = tf.image.resize(tensor_input, [int(resize_range*aspect_ratio), resize_range])
    #print('Label Shape:', label.size)
    resized_label = tf.image.resize(tensor_label, [int(resize_range*aspect_ratio), resize_range])
    
    p_transform = random.random() # Generate a random number between 0.0 and 1.0

    input = tensor_to_image(resized_input)
    label = tensor_to_image(resized_label)

    if mode == 'train' and p_transform <= augmentation_prob:
            
        # Generates 1 of 4 possible rotation degree possibilities

        rotation_possibilities =[0, 90, 180, 270]
        rotation_option = random.randint(0,3)
        rotation_degree = rotation_possibilities[rotation_option]

        if rotation_degree==90 or rotation_degree==270:
            aspect_ratio = 1/aspect_ratio

        random_rot1 = random.randint(-rotation_degree, rotation_degree)
        if random_rot1 is not 0 or 360:
            print("randomrot1:", random_rot1)
            print("input=", input)
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

        label = np.expand_dims(label_img, axis=0)
        print('Label shape', label.shape )
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
    input = []
    label = []
    for file_name in os.listdir(input_path):
        input_img = tf.keras.preprocessing.image.load_img(input_path + '/' + file_name)
        label_img = tf.keras.preprocessing.image.load_img(label_path + '/' + file_name[:-len(".jpg")] + '_Segmentation.png')
        processed_input, processed_label = getInputLabel(input_img, label_img, mode=mode, augmentation_prob=augmentation_prob)
        input.append(processed_input)
        label.append(processed_label)
    return input, label
