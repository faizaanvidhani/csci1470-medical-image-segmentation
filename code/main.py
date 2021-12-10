from solver import Solver
from preprocess import get_data

def main():
    train_inputs, train_labels = get_data('../data/ISBI2016_ISIC_Part1_Training_Data', '../data/ISBI2016_ISIC_Part1_Training_GroundTruth', mode='train')
    test_inputs, test_labels = get_data('../data/ISBI2016_ISIC_Part1_Test_Data', '../data/ISBI2016_ISIC_Part1_Test_GroundTruth', mode='test',augmentation_prob=0.)
    
    solver = Solver(train_inputs, train_labels, test_inputs, test_labels)
    solver.train()
    solver.test()
    solver.visualize_loss()
    solver.visualize_accuracy()

if __name__ == '__main__':
    main()
    