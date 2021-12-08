import tensorflow as tf
import argparse
import os
from solver import Solver
from preprocess import get_data
#from torch.backends import cudnn
import random


def main():
    #cudnn.benchmark = True
    """ if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return """

    # Create directories if not exist
    """ if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = random.choice([100,150,200,250])
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config) """

    # New Way of Importing Train Data and Test Data
    '''
    train_inputs, train_labels = get_data('../data/ISBI2016_ISIC_Part1_Training_Data', '../data/ISBI2016_ISIC_Part1_Training_GroundTruth', 900, mode='train')
    test_inputs, test_labels = get_data('../data/ISBI2016_ISIC_Part1_Test_Data', '../data/ISBI2016_ISIC_Part1_Test_GroundTruth', 379, mode='test',augmentation_prob=0.)
    '''
    """
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
     valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.) 
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    """
    train_inputs, train_labels = get_data('../data/ISBI2016_ISIC_Part1_Training_Data', '../data/ISBI2016_ISIC_Part1_Training_GroundTruth', 900, mode='train')
    #test_inputs, test_labels = get_data('../data/ISBI2016_ISIC_Part1_Test_Data', '../data/ISBI2016_ISIC_Part1_Test_GroundTruth', 379, mode='test',augmentation_prob=0.)
    
    # Train and sample the images
    #trainex = tf.zeros([336, 256, 3])
    #labelex = tf.zeros([336, 256, 3])
   # solver = Solver(trainex,labelex,None,None)
    #solver.train()
    
    solver = Solver(train_inputs, train_labels, None, None)
    solver.train()
    #solver.test()






if __name__ == '__main__':
    main()
    