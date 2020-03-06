import keras
import skimage.transform as tf
import numpy as np
import random
import scipy


class BaseGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, input_shape, augment):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.augment = augment

        self._mid = input_shape[0] // 2
        # initialize augmentation ranges
        self.translations = [f for f in range(-3, 4)]
        self.rotations = ([f for f in range(330, 365, 5)]
                          + [f for f in range(0, 35, 5)])
        super().__init__()

    def _augment_img(self, img, x_trans, y_trans, angle, flip=False):
        if flip:
            img = np.fliplr(img)
        trans = tf.AffineTransform(translation=[x_trans, y_trans])
        img = tf.rotate(img, angle, center=(self._mid, self._mid))
        return tf.warp(img, trans, preserve_range=True)

    def _augment(self, img):
        """Add the same random rotation and translation, chosen from the
        predefined ranges, to both given images"""
        # choose random variation
        x_trans = random.sample(self.translations, 1)[0]
        y_trans = random.sample(self.translations, 1)[0]
        angle = random.sample(self.rotations, 1)[0]

        return self._augment_img(img, x_trans, y_trans, angle)

    def __next__(self):
        inputs, targets = self.__getitem__(None)
        return inputs, targets


class SiameseGenerator(BaseGenerator):

    def __init__(self, distances, images, mapping, batch_size, input_shape,
                 augment=True):
        self.distances = distances
        self.images = np.copy(images)
        self.mapping = mapping
        self.rotations = [f for f in range(0, 365, 5)]

        super().__init__(batch_size, input_shape, augment)

    def __len__(self):
        # number of batches to cover all possible image pairs
        return int(np.ceil(scipy.special.comb(len(self.images), 2)
                           / self.batch_size))

    def _augment_pair(self, img1, img2):
        # choose random variation
        x_trans = random.sample(self.translations, 1)[0]
        y_trans = random.sample(self.translations, 1)[0]
        angle = random.sample(self.rotations, 1)[0]

        flip = random.sample([True, False], 1)[0]

        i1 = self._augment_img(img1, x_trans, y_trans, angle, flip)
        i2 = self._augment_img(img2, x_trans, y_trans, angle, flip)
        return i1, i2

    def __getitem__(self, idx):
        pairs = [np.zeros((self.batch_size, *self.input_shape))
                 for i in range(2)]
        targets = np.zeros((self.batch_size,))

        g_is = random.sample(self.mapping.keys(), self.batch_size)
        g_js = random.sample(self.mapping.keys(), self.batch_size)
        for i, (g_i, g_j) in enumerate(zip(g_is, g_js)):
            # extract distance from distance matrix & get corresponding images
            targets[i] = self.distances[g_i, g_j]
            if self.augment:
                i1, i2 = self._augment_pair(self.images[self.mapping[g_i]],
                                            self.images[self.mapping[g_j]])
            else:
                i1 = self.images[self.mapping[g_i]]
                i2 = self.images[self.mapping[g_j]]
            pairs[0][i] = i1
            pairs[1][i] = i2

        return pairs, targets


class KerasGenerator(BaseGenerator):

    def __init__(self, src_generator, batch_size, input_shape, augment=False):
        self.generator = src_generator

        super().__init__(batch_size, input_shape, augment)

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, idx):
        inputs = np.zeros((self.batch_size, *self.input_shape))

        batch = next(self.generator)
        if self.augment:
            for i in range(self.batch_size):
                inputs[i] = self._augment(batch[0][i])
        else:
            inputs = batch[0]

        return inputs, batch[1]
