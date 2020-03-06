import keras


def main(name, test_path, model_path):

    batch_size = 10
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
                                                            rescale=1./255)
    test_gen = test_datagen.flow_from_directory(test_path,
                                                batch_size=batch_size,
                                                target_size=(28, 28),
                                                color_mode="grayscale")

    model = keras.models.load_model(model_path)

    score = model.evaluate_generator(test_gen, verbose=0, steps=len(test_gen))

    print('{},{}'.format(name, score[1]))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate character '
                                     'recognition performance of a trained '
                                     'digit recognizer model.')
    parser.add_argument('name', help='Run name.')
    parser.add_argument('test_path', help='Path to folder structure '
                        'containing the test images.')
    parser.add_argument('model_path', help='Path to the trained digit '
                        'recognizer.')

    args = vars(parser.parse_args())
    main(**args)
