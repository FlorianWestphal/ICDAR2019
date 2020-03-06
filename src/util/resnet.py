import keras


class ResNet:
    # implementation taken from
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py

    def __init__(self, input_shape, num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self._num_filters = 16
        self._num_res_blocks = 3

    def setup(self, model=None):
        if model:
            inputs, base = self._load_base(model)
            self.model = self._load_model(inputs, base, model)
        else:
            inputs, base = self._build_base()
            self.model = self._build_model(inputs, base)
        self.base = self._build_base_model(inputs, base)

    def load_from_base(self, base, trainable_base=False):
        # third layer of siamese model is resnet base model
        model = base.layers[2]
        inputs, base = self._load_base(model)
        self.base = self._build_base_model(inputs, base)

        if not trainable_base:
            for layer in self.base.layers:
                layer.trainable = False
        recognizer = self.base(inputs)

        outputs = keras.layers.Dense(
                                self.num_classes,
                                activation='softmax',
                                kernel_initializer='he_normal')(recognizer)

        self.model = keras.models.Model(inputs=inputs, outputs=outputs)

    def _resnet_layer(self, inputs, num_filters=16, kernel_size=3, strides=1,
                      activation='relu', batch_normalization=True,
                      conv_weights=None, bn_weights=None):
        """2D Convolution-Batch Normalization-Activation stack builder
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = keras.layers.Conv2D(
                            num_filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            weights=conv_weights)

        x = inputs
        x = conv(x)
        if batch_normalization:
            x = keras.layers.BatchNormalization(weights=bn_weights)(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)

        return x

    def _load_base(self, model):
        # Start model definition.
        num_filters = self._num_filters

        idx = 0
        layers = model.layers

        inputs = keras.layers.Input(shape=self.input_shape)

        idx += 1        # skip over input layer
        x = self._resnet_layer(inputs=inputs,
                               conv_weights=layers[idx].get_weights(),
                               bn_weights=layers[idx + 1].get_weights())

        idx += 3        # skip activation layer weights, since there are no
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(self._num_res_blocks):
                strides = 1
                # first layer but not first stack
                if stack > 0 and res_block == 0:
                    strides = 2  # downsample
                    # batch norm layer is defined after conv layer added to
                    # fix dimensions
                    idx_step = 1
                else:
                    idx_step = 0
                y = self._resnet_layer(
                                    inputs=x,
                                    num_filters=num_filters,
                                    strides=strides,
                                    conv_weights=layers[idx].get_weights(),
                                    bn_weights=layers[idx+1].get_weights())
                idx += 3
                y = self._resnet_layer(
                        inputs=y,
                        num_filters=num_filters,
                        activation=None,
                        conv_weights=layers[idx].get_weights(),
                        bn_weights=layers[idx + 1 + idx_step].get_weights())
                # first layer but not first stack
                if stack > 0 and res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    # this causes following layer order:
                    # conv(3) conv(1) batch_norm add --> we have to load
                    # conv(3) and batch_norm together, thus skip conv(1)
                    idx += 1
                    x = self._resnet_layer(
                                    inputs=x,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False,
                                    conv_weights=layers[idx].get_weights())

                # only increase layer index by 2 since there is not activation
                # layer here
                idx += 2

                x = keras.layers.add([x, y])
                x = keras.layers.Activation('relu')(x)
                # skip add and activation layer, since they do not have
                # weights
                idx += 2
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = keras.layers.AveragePooling2D(pool_size=7)(x)
        y = keras.layers.Flatten()(x)

        return inputs, y

    def _build_base(self):
        # Start model definition.
        num_filters = self._num_filters

        inputs = keras.layers.Input(shape=self.input_shape)
        x = self._resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(self._num_res_blocks):
                strides = 1
                # first layer but not first stack
                if stack > 0 and res_block == 0:
                    strides = 2  # downsample
                y = self._resnet_layer(inputs=x,
                                       num_filters=num_filters,
                                       strides=strides)
                y = self._resnet_layer(inputs=y,
                                       num_filters=num_filters,
                                       activation=None)
                # first layer but not first stack
                if stack > 0 and res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self._resnet_layer(inputs=x,
                                           num_filters=num_filters,
                                           kernel_size=1,
                                           strides=strides,
                                           activation=None,
                                           batch_normalization=False)
                x = keras.layers.add([x, y])
                x = keras.layers.Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = keras.layers.AveragePooling2D(pool_size=7)(x)
        y = keras.layers.Flatten()(x)

        return inputs, y

    def _load_model(self, inputs, base, model):
        layers = model.layers
        outputs = keras.layers.Dense(
                                    self.num_classes,
                                    activation='softmax',
                                    kernel_initializer='he_normal',
                                    weights=layers[-1].get_weights())(base)

        # Instantiate model.
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def _build_model(self, inputs, base):
        """ResNet Version 1 Model builder [a]
        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved
        (downsampled) by a convolutional layer with strides=2, while the
        number of filters is doubled. Within each stage, the layers have the
        same number filters and the same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M
        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)
        # Returns
            model (Model): Keras model instance
        """

        outputs = keras.layers.Dense(self.num_classes,
                                     activation='softmax',
                                     kernel_initializer='he_normal')(base)

        # Instantiate model.
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def _build_base_model(self, inputs, base):
        return keras.models.Model(inputs=inputs, outputs=base)


class SiameseResNet(ResNet):

    # workaround to make it possible to load stored model with lambda layer
    def _lambda_distance(self, tensors):
        import keras
        return keras.backend.abs(tensors[0] - tensors[1])

    def setup(self, model=None):
        super().setup(model)

        left_input = keras.layers.Input(self.input_shape)
        right_input = keras.layers.Input(self.input_shape)

        encoded_l = self.base(left_input)
        encoded_r = self.base(right_input)

        L1_layer = keras.layers.Lambda(self._lambda_distance)
        L1_distance = L1_layer([encoded_l, encoded_r])

        prediction = keras.layers.Dense(1, activation='linear')(L1_distance)
        self.siamese_net = keras.models.Model(
                                            inputs=[left_input, right_input],
                                            outputs=prediction)
