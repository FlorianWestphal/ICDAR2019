import keras
import os
import pandas as pd

import util.generator as generator
import util.resnet as resnet


def new_model(input_shape):
    model = resnet.ResNet(input_shape)
    model.setup()
    model = model.model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


def main(name, train_path, valid_path, log_path):
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
                                                            rescale=1./255)
    batch_size = 10
    epochs = 50
    input_shape = (28, 28, 1)

    validation_generator = test_datagen.flow_from_directory(
                                                        valid_path,
                                                        batch_size=batch_size,
                                                        target_size=(28, 28),
                                                        color_mode="grayscale")

    model = new_model(input_shape)

    model_path = os.path.join(log_path, '{}.h5'.format(name))
    save_callback = keras.callbacks.ModelCheckpoint(model_path,
                                                    save_best_only=True)

    # initialize train generator for current run
    train_generator = test_datagen.flow_from_directory(train_path,
                                                       batch_size=batch_size,
                                                       target_size=(28, 28),
                                                       color_mode="grayscale")
    # allow for augmentation
    gen = generator.KerasGenerator(train_generator, batch_size, input_shape,
                                   augment=True)

    # train model --> set steps_per_epoch to fixed value to ensure same number
    # of update steps regardless training set size
    history = model.fit_generator(
                                gen,
                                steps_per_epoch=500,
                                epochs=epochs,
                                validation_data=validation_generator,
                                validation_steps=len(validation_generator),
                                verbose=0,
                                callbacks=[save_callback])

    # store training process
    pd.DataFrame.from_dict(history.history).to_csv(
                                        os.path.join(log_path,
                                                     '{}.csv'.format(name)),
                                        index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train digit recognizer '
                                     'from scratch.')
    parser.add_argument('name', help='Name of this run to be used in log '
                        'files and model files.')
    parser.add_argument('train_path', help='Path to the folder structure '
                        'containing the character images of the training '
                        'set.')
    parser.add_argument('valid_path', help='Path to the folder structure '
                        'containing the character images of the validation '
                        'set.')
    parser.add_argument('log_path', help='Path ot the folder in which the '
                        'trained model and the log file of the training '
                        'process should be saved.')

    args = vars(parser.parse_args())
    main(**args)
