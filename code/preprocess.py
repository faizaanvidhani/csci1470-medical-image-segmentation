import random
from random import shuffle
import os
import numpy as np
import tensorflow as tf
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

        #pytorch Transform = []
        resize_range = random.randint(300,320)

        tf.image.resize
        p_transform = random.random() # Generate a random number between 0.0 and 1.0










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