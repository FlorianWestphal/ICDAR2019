import keras
import numpy as np
import os
import pandas as pd

from datetime import datetime

import util.generator as generator
import util.resnet as resnet

BATCH_SIZE = 10
INPUT_SHAPE = (28, 28, 1)


def load_graphs(graph_path):
    data = np.load(graph_path)
    return data['distances'], data['labels'], data['paths']


def load_imgs(img_path):
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    gen = test_datagen.flow_from_directory(
                                img_path,
                                batch_size=BATCH_SIZE,
                                target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                color_mode="grayscale",
                                shuffle=False)

    paths = [f.split('.')[0] for f in gen.filenames]
    img_number = len(paths)

    # extract images from generator in order to allow matching against loaded
    # graphs
    images = np.zeros((img_number, *INPUT_SHAPE))
    labels = np.zeros((img_number, 10))
    for i, batch in enumerate(gen):
        idx = i * BATCH_SIZE
        images[idx:idx + BATCH_SIZE, :, :, :1] = batch[0]
        labels[idx:idx + BATCH_SIZE] = batch[1]
        if idx+BATCH_SIZE >= img_number:
            break

    return images, labels, paths


def map_graphs_imgs(g_paths, i_paths):
    mapping = {}
    for i, g in enumerate(g_paths):
        for j, img in enumerate(i_paths):
            if g == img:
                mapping[i] = j
    return mapping


def load_data(graph_path, img_path):
    distances, g_labels, g_paths = load_graphs(graph_path)
    images, i_labels, i_paths = load_imgs(img_path)
    g_i_mapping = map_graphs_imgs(g_paths, i_paths)

    return images, distances, g_i_mapping, g_labels


def main(name, train_graph_path, train_img_path, valid_graph_path,
         valid_img_path, model_path, log_path):
    # data is a tuple containing: (images, distances, mapping, graph labels)
    print('load train {}'.format(datetime.now().isoformat(timespec='seconds')))
    train_data = load_data(train_graph_path, train_img_path)
    print('load test {}'.format(datetime.now().isoformat(timespec='seconds')))
    valid_data = load_data(valid_graph_path, valid_img_path)

    train_generator = generator.SiameseGenerator(train_data[1], train_data[0],
                                                 train_data[2], BATCH_SIZE,
                                                 INPUT_SHAPE)
    valid_generator = generator.SiameseGenerator(valid_data[1], valid_data[0],
                                                 valid_data[2], BATCH_SIZE,
                                                 INPUT_SHAPE, augment=False)

    epochs = 100

    # load model if specified
    if model_path is not None:
        model = keras.models.load_model(model_path)
    else:
        model = None
    siamese_net = resnet.SiameseResNet(INPUT_SHAPE)
    siamese_net.setup(model)

    siamese_net.siamese_net.compile(loss=keras.losses.mean_squared_error,
                                    optimizer=keras.optimizers.Adadelta())

    dest_model_path = os.path.join(log_path, 'graphtrain_{}.h5'.format(name))
    save_callback = keras.callbacks.ModelCheckpoint(dest_model_path,
                                                    save_best_only=True)

    print('start train {}'.format(datetime.now()
                                  .isoformat(timespec='seconds')))
    history = siamese_net.siamese_net.fit_generator(
                                        train_generator,
                                        epochs=epochs,
                                        steps_per_epoch=len(train_generator),
                                        validation_data=valid_generator,
                                        validation_steps=len(valid_generator),
                                        verbose=0,
                                        callbacks=[save_callback])

    # store training process
    pd.DataFrame.from_dict(history.history).to_csv(
                               os.path.join(log_path,
                                            'graphtrain_{}.csv'.format(name)),
                               index=False)

    print('end train {}'.format(datetime.now().isoformat(timespec='seconds')))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
                    description='Train Siamese network to predict HED between '
                    'graphs of the input images.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('name', help='Name of this run to be used in log '
                        'files and model files.')
    parser.add_argument('train_graph_path', help='Path to the NPZ file with '
                        'the edit distances of the training set graphs.')
    parser.add_argument('train_img_path', help='Path to the folder structure '
                        'containing the character images of the training set.')
    parser.add_argument('valid_graph_path', help='Path to the NPZ file with '
                        'the edit distances of the validation set graphs.')
    parser.add_argument('valid_img_path', help='Path to the folder structure '
                        'containing the character images of the validation '
                        'set.')
    parser.add_argument('log_path', help='Path ot the folder in which the '
                        'trained model and the log file of the training '
                        'process should be saved.')
    parser.add_argument('--model_path', help='Path to pre-trained Siamese '
                        'network model.', default=None)

    args = vars(parser.parse_args())
    main(**args)
