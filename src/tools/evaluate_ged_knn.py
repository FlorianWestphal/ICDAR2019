import numpy as np
import os

import util.knn as knn


def make_train(train_path, paths):
    # list all files in base directory and normalize their path name
    selected = [os.path.join(str(i), f.split('.')[0]) for i in range(10)
                for f in os.listdir(os.path.join(train_path, str(i)))]

    return [np.where(paths == i)[0][0] for i in selected]


def main(name, train_path, dist_path):

    data = np.load(dist_path)
    distances = data['distances']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    train_paths = data['train_paths']

    classifier = knn.KNearestNeighborsTrainTest(distances, train_labels,
                                                test_labels)

    train = make_train(train_path, train_paths)
    acc, _ = classifier.score(train)

    print('{},{}'.format(name, acc))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate character '
                                     'recognition performance of a kNN based '
                                     'approach using the pre-computed HEDs.')
    parser.add_argument('name', help='Run name.')
    parser.add_argument('train_path', help='Path to folder structure '
                        'containing the training images.')
    parser.add_argument('dist_path', help='Path to the NPZ file containing '
                        'the pre-computed HEDs between the graphs of the '
                        'training and the test set.')

    args = vars(parser.parse_args())
    main(**args)
