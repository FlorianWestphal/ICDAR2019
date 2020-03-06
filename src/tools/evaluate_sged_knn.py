import keras
import numpy as np
import sklearn.neighbors

import util.generator as generator
import util.resnet as resnet


def load_model(model_path):
    m = resnet.SiameseResNet((28, 28, 1))
    loaded = resnet.ResNet((28, 28, 1))
    custom = {'_lambda_distance': m._lambda_distance}
    model = keras.models.load_model(model_path,
                                    custom_objects=custom)
    loaded.load_from_base(model)

    return loaded


def make_train(model, train_path, gen, batch_size):
    train_generator = gen.flow_from_directory(
                        train_path,
                        batch_size=batch_size,
                        target_size=(28, 28),
                        color_mode="grayscale")
    keras_gen = generator.KerasGenerator(train_generator,
                                         batch_size=batch_size,
                                         input_shape=(28, 28, 1),
                                         augment=True)

    size = min(91 * train_generator.samples, 50000)
    train = np.zeros((size, 64))
    train_label = np.zeros(size)
    i = 0
    while i < size:
        for batch in keras_gen:
            train[i:i+batch_size] = model.base.predict(batch[0])
            train_label[i:i+batch_size] = np.argmax(np.array(batch[1]), axis=1)
            i += batch_size
            if i == size:
                break
    return train, train_label, train_generator.samples


def make_test(model, test_path, gen, batch_size):
    test_gen = gen.flow_from_directory(
                                test_path,
                                batch_size=batch_size,
                                target_size=(28, 28),
                                color_mode="grayscale")

    test = np.zeros((len(test_gen), 64))
    test_labels = np.zeros(len(test_gen))
    i = 0
    for batch in test_gen:
        test[i:i+batch_size, ] = model.base.predict(batch[0])
        test_labels[i:i+batch_size] = np.argmax(np.array(batch[1]), axis=1)
        i += batch_size
        if i == len(test_gen):
            break

    return test, test_labels


def main(name, train_path, test_path, model_path):

    model = load_model(model_path)

    batch_size = 10
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
                                                            rescale=1./255)

    test, test_labels = make_test(model, test_path, test_datagen, batch_size)
    train, train_label, samples = make_train(model, train_path, test_datagen,
                                             batch_size)
    # choose k to be at most as large as supported by the training dataset
    # or as configured, if enough training samples are available
    k = min(samples//10, 10)
    neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train, train_label)

    acc = neigh.score(test, test_labels)

    print('{},{}'.format(name, acc))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate character '
                                     'recognition performance of a kNN based '
                                     'approach using the embedding vector '
                                     'produced by the pre-trained Siamese '
                                     'network.')
    parser.add_argument('name', help='Run name.')
    parser.add_argument('train_path', help='Path to folder structure '
                        'containing the training images.')
    parser.add_argument('test_path', help='Path to the folder structure '
                        'containing the test images.')
    parser.add_argument('model_path', help='Path to the pre-trained Siamese '
                        'network.')

    args = vars(parser.parse_args())
    main(**args)
