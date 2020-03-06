import keras
import os
import pandas as pd

from datetime import datetime

import util.resnet as resnet
import util.generator as generator


INPUT_SHAPE = (28, 28, 1)
BATCH_SIZE = 10


def load_data(datagen, path):
    # load data
    generator = datagen.flow_from_directory(
                                path,
                                batch_size=BATCH_SIZE,
                                target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                color_mode="grayscale")
    return generator


def load_model(model_path, trainable):
    # load model
    m = resnet.SiameseResNet(INPUT_SHAPE)
    custom = {'_lambda_distance': m._lambda_distance}
    model = keras.models.load_model(model_path, custom_objects=custom)
    recognizer = resnet.ResNet(INPUT_SHAPE)
    recognizer.load_from_base(model, trainable_base=trainable)
    recognizer.model.compile(loss=keras.losses.categorical_crossentropy,
                             optimizer=keras.optimizers.Adadelta(),
                             metrics=['accuracy'])
    return recognizer


def main(name, model_path, train_path, valid_path, log_path, trainable):

    print('start {}'.format(datetime.now().isoformat(timespec='seconds')))
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = load_data(test_datagen, train_path)
    # allow for augmentation
    train_gen = generator.KerasGenerator(train_generator, BATCH_SIZE,
                                         INPUT_SHAPE, augment=True)

    valid_generator = load_data(test_datagen, valid_path)

    recognizer = load_model(model_path, trainable)

    epochs = 50
    save_model_path = os.path.join(log_path, 'digit_{}.h5'.format(name))
    save_callback = keras.callbacks.ModelCheckpoint(save_model_path,
                                                    save_best_only=True)

    print('start train {}'.format(datetime.now()
                                  .isoformat(timespec='seconds')))
    history = recognizer.model.fit_generator(
                                        train_gen,
                                        epochs=epochs,
                                        steps_per_epoch=500,
                                        validation_data=valid_generator,
                                        validation_steps=len(valid_generator),
                                        verbose=0,
                                        callbacks=[save_callback])

    print('end train {}'.format(datetime.now().isoformat(timespec='seconds')))
    # store training process
    pd.DataFrame.from_dict(history.history).to_csv(
                                    os.path.join(log_path,
                                                 'digit_{}.csv'.format(name)),
                                    index=False)

    print('finish {}'.format(datetime.now().isoformat(timespec='seconds')))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train digit recognizer '
                                     'based on provided pre-trained Siamese '
                                     'network.')
    parser.add_argument('name', help='Name of this run to be used in log '
                        'files and model files.')
    parser.add_argument('model_path', help='Path to the file containing the '
                        'pre-trained Siamese network.')
    parser.add_argument('train_path', help='Path to the folder structure '
                        'containing the character images of the training '
                        'set.')
    parser.add_argument('valid_path', help='Path to the folder structure '
                        'containing the character images of the validation '
                        'set.')
    parser.add_argument('log_path', help='Path ot the folder in which the '
                        'trained model and the log file of the training '
                        'process should be saved.')
    parser.add_argument('--trainable', help='Indicate if the CNN base should '
                        'be re-trained.', action='store_true')

    args = vars(parser.parse_args())
    main(**args)
