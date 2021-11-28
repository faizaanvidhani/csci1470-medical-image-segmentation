import random
from random import shuffle
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image.rotate as rotate
import tensorflow.keras.layers.CenterCrop as crop
from PIL import Image

class ImageFolder(data.Dataset):
    def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.4):
        self.root = root
        self.label_paths = root[:-1]+'_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root,x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.rotation = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob

    def get_item(self, index):
        input_path = self.image_paths[index]
        filename = input_path.split('_')[-1][:-len(".jpg")]
        label_path = self.label_paths + 'ISIC_' + filename + '_segmentation.png'

        input = Image.open(input_path)
        label = Image.open(label_path)

        aspect_ratio = input.size[1]/input.size[0]
        resize_range = random.randint(300,320)

        input = tf.image.resize(input, [int(resize_range*aspect_ratio), resize_range])
        label = tf.image.resize(label, [int(resize_range*aspect_ratio), resize_range])
        
        p_transform = random.random() # Generate a random number between 0.0 and 1.0

        if self.mode == 'train' and p_transform <= self.augmentation_prob:
             
             # Generates 1 of 4 possible rotation degree possibilities
            rotation_option = random.randint(0,3)
            rotation_degree = self.rotation[rotation_option]

            if rotation_degree==90 or rotation_degree==270:
                aspect_ratio = 1/aspect_ratio

            random_rot1 = random.randint(-rotation_degree, rotation_degree)
            input = rotate(random_rot1)
            label = rotate(random_rot1)

            random_rot2 = random.randint(-10,10)
            input = rotate(random_rot2)
            label = rotate(random_rot2)

            random_crop = random.randint(250,270)
            input = crop(input, int(random_crop*aspect_ratio), random_crop)
            label = crop(label, int(random_crop*aspect_ratio), random_crop)

            shift_left = random.randint(0,20)
            shift_up = random.randint(0,20)
            shift_right = input.size[0] - random.randint(0,20)
            shift_down = input.size[1] - random.randint(0,20)
            
            input = input.crop(box=(shift_left, shift_up, shift_right, shift_down))
            label = label.crop(box=(shift_left, shift_up, shift_right, shift_down))

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

        # Converting images to tensor

        input = tf.image.convert_image_dtype(input, dtype=tf.float32)
        label = tf.image.convert_image_dtype(label, dtype=tf.float32)

        tf.image.per_image_standardization(input)

        return input, label

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
	return data_loader







# My Implementation

def get_data(inputs_file_path, labels_file_path, num_examples):
    # num_examples = 900 for train files and 300 for test files

    for i in range(num_examples):
        rotation = [0, 90, 180, 270, 360]
        input = 
        label = 

        # Normalize Image but not GT?
        return input, label


# Note: Image dimensions are 1022 × 767
for filename in os.listdir(directory):