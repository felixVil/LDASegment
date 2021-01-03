from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Concatenate
from keras.applications.vgg19 import VGG19
from keras.models import Model
import keras.backend as K
from NetworkLayers.BilinearUpSampling import BilinearUpSampling2D
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201, preprocess_input
import numpy as np


class VGGDeconv:

    TOP_LAYERS_NAMES = ['conv_1x1_stacked_1', 'conv_1x1_stacked_2', 'conv_1x1_1_final']
    ARCHITECTURE_DICT = {'feature_maps': [64, 128, 256, 0, 0], 'deconv_factors': [2, 4, 8, 16, 32]}

    def __init__(self, feature_map_nums, img_input_shape, use_lda=True):
        self.use_lda = use_lda
        print('image data format:' + K.image_data_format())
        self.filters_blocks = feature_map_nums
        self.implement_blocks = [self.filters_blocks[k] > 0 for k in range(len(self.filters_blocks))]
        self.img_input_shape = img_input_shape
        self.block_outputs, self.outputs_to_concat = ([] for _ in range(2))
        self.deconv_factors = np.array([2, 4, 8, 16, 32])
        self._single_template_net()
        self._assign_pretrained_weights()

    def _block1_implement(self, img_input, is_trainable=False):
        # Block 1
        if self.implement_blocks[0]:
            x = Conv2D(self.filters_blocks[0], (3, 3), activation='relu', padding='same', name='block1_conv1',
                       trainable = is_trainable)(img_input)
            x = Conv2D(self.filters_blocks[0], (3, 3), activation='relu', padding='same', name='block1_conv2',
                       trainable = is_trainable)(x)
            self.block_outputs.append(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                                                   trainable = is_trainable)(x))

    def _block2_implement(self, is_trainable=False):
        # Block 2
        if self.implement_blocks[1]:
            x = Conv2D(self.filters_blocks[1], (3, 3), activation='relu', padding='same', name='block2_conv1', trainable = is_trainable)(self.block_outputs[0])
            x = Conv2D(self.filters_blocks[1], (3, 3), activation='relu', padding='same', name='block2_conv2', trainable = is_trainable)(x)
            self.block_outputs.append(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', trainable = is_trainable)(x))

    def _block3_implement(self, is_trainable=False):
        # Block 3
        if self.implement_blocks[2]:
            x = Conv2D(self.filters_blocks[2], (3, 3), activation='relu', padding='same', name='block3_conv1', trainable = is_trainable)(self.block_outputs[1])
            x = Conv2D(self.filters_blocks[2], (3, 3), activation='relu', padding='same', name='block3_conv2', trainable = is_trainable)(x)
            x = Conv2D(self.filters_blocks[2], (3, 3), activation='relu', padding='same', name='block3_conv3', trainable = is_trainable)(x)
            x = Conv2D(self.filters_blocks[2], (3, 3), activation='relu', padding='same', name='block3_conv4', trainable = is_trainable)(x)
            self.block_outputs.append(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable = is_trainable)(x))

    def _block4_implement(self, is_trainable=False):
        # Block 4
        if self.implement_blocks[3]:
            x = Conv2D(self.filters_blocks[3], (3, 3), activation='relu', padding='same', name='block4_conv1', trainable = is_trainable)(self.block_outputs[2])
            x = Conv2D(self.filters_blocks[3], (3, 3), activation='relu', padding='same', name='block4_conv2', trainable = is_trainable)(x)
            x = Conv2D(self.filters_blocks[3], (3, 3), activation='relu', padding='same', name='block4_conv3', trainable = is_trainable)(x)
            x = Conv2D(self.filters_blocks[3], (3, 3), activation='relu', padding='same', name='block4_conv4', trainable = is_trainable)(x)
            self.block_outputs.append(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable = is_trainable)(x))

    def _block5_implement(self, is_trainable=False):
        # Block 5
        if self.implement_blocks[4]:
            x = Conv2D(self.filters_blocks[4], (3, 3), activation='relu', padding='same', name='block5_conv1', trainable = is_trainable)(self.block_outputs[3])
            x = Conv2D(self.filters_blocks[4], (3, 3), activation='relu', padding='same', name='block5_conv2', trainable = is_trainable)(x)
            x = Conv2D(self.filters_blocks[4], (3, 3), activation='relu', padding='same', name='block5_conv3', trainable = is_trainable)(x)
            x = Conv2D(self.filters_blocks[4], (3, 3), activation='relu', padding='same', name='block5_conv4', trainable = is_trainable)(x)
            self.block_outputs.append(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', trainable = is_trainable)(x))

    def _deconv_blocks_implement(self):
        for k in range(len(self.implement_blocks)):
        # Deconv of block k
            if self.implement_blocks[k]:
                self.outputs_to_concat.append(BilinearUpSampling2D(size=(self.deconv_factors[k], self.deconv_factors[k]))(self.block_outputs[k]))

    def _single_template_net(self):
        img_input = Input(shape=self.img_input_shape)
        self._block1_implement(img_input)
        self._block2_implement()
        self._block3_implement()
        self._block4_implement()
        self._block5_implement()
        self._deconv_blocks_implement()

        if K.image_data_format() == 'channels_first':
            concat_axis = 0
        else:
            concat_axis = -1
        if len(self.outputs_to_concat) > 1:
            merged_deconv_layers = Concatenate(axis=concat_axis, name='net_feature_map')(self.outputs_to_concat)
        else:
            merged_deconv_layers = self.outputs_to_concat
        self.backend_model = Model(inputs=img_input, outputs=merged_deconv_layers)

    def _assign_pretrained_weights(self):
        # obtaining pre-trained vgg19 model.
        original_vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=self.img_input_shape)

        # getting weights from specific layers
        layer_dict = dict([(layer.name, layer) for layer in original_vgg_model.layers])
        needed_layers = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2',
                         'block3_conv3', 'block3_conv4', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4',
                         'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4']
        weights = {}
        for layer in needed_layers:
            weights[layer] = layer_dict[layer].get_weights()

        layer_dict_new = dict([(layer.name, layer) for layer in self.backend_model.layers])
        if self.implement_blocks[0]:
            layer_dict_new['block1_conv1'].set_weights(weights['block1_conv1'])
            layer_dict_new['block1_conv2'].set_weights(weights['block1_conv2'])

        if self.implement_blocks[1]:
            layer_dict_new['block2_conv1'].set_weights(weights['block2_conv1'])
            layer_dict_new['block2_conv2'].set_weights(weights['block2_conv2'])

        if self.implement_blocks[2]:
            layer_dict_new['block3_conv1'].set_weights(weights['block3_conv1'])
            layer_dict_new['block3_conv2'].set_weights(weights['block3_conv2'])
            layer_dict_new['block3_conv3'].set_weights(weights['block3_conv3'])
            layer_dict_new['block3_conv4'].set_weights(weights['block3_conv4'])

        if self.implement_blocks[3]:
            layer_dict_new['block4_conv1'].set_weights(weights['block4_conv1'])
            layer_dict_new['block4_conv2'].set_weights(weights['block4_conv2'])
            layer_dict_new['block4_conv3'].set_weights(weights['block4_conv3'])
            layer_dict_new['block4_conv4'].set_weights(weights['block4_conv4'])

        if self.implement_blocks[4]:
            layer_dict_new['block5_conv1'].set_weights(weights['block5_conv1'])
            layer_dict_new['block5_conv2'].set_weights(weights['block5_conv2'])
            layer_dict_new['block5_conv3'].set_weights(weights['block5_conv3'])
            layer_dict_new['block5_conv4'].set_weights(weights['block5_conv4'])

    def predict_backend(self, new_sample):
        new_sample = new_sample[:, :, :, ::-1]
        backend_tensor = self.backend_model.predict(new_sample, 1)
        backend_tensor = np.squeeze(backend_tensor)
        return backend_tensor

class DenseNetDeconv:

    ARCHITECTURE_DICT = {'DenseNet121': {'feature_maps': [256, 512, 1024, 1024], 'layers':[48, 136, 308, 424], 'deconv_factors': [4, 8, 16, 32]},
                        'DenseNet169': {'feature_maps': [256, 512, 1280, 1664], 'layers': [48, 136, 364, 592], 'deconv_factors': [4, 8, 16, 32]},
                        'DenseNet201': {'feature_maps': [256, 512, 1792, 1920], 'layers': [48, 136, 476, 704], 'deconv_factors': [4, 8, 16, 32]}}

    def __init__(self, img_input_shape, densenet_name, num_of_blocks):
        assert densenet_name in DenseNetDeconv.ARCHITECTURE_DICT.keys()
        self.densenet_name  = densenet_name
        if K.image_data_format() == 'channels_first':
            concat_axis = 0
        else:
            concat_axis = -1
        if densenet_name == 'DenseNet121':
            self.densenet_model = DenseNet121(include_top=False, weights='imagenet',input_tensor=None,
                                              input_shape=img_input_shape, pooling=None, classes=1000)
        elif densenet_name == 'DenseNet169':
            self.densenet_model = DenseNet169(include_top=False, weights='imagenet', input_tensor=None,
                                              input_shape=img_input_shape, pooling=None, classes=1000)
        else:
            self.densenet_model = DenseNet201(include_top=False, weights='imagenet', input_tensor=None,
                                              input_shape=img_input_shape, pooling=None, classes=1000)

        layers_to_apply_deconv = DenseNetDeconv.ARCHITECTURE_DICT[self.densenet_name]['layers']


        self.deconv_factors = np.array(DenseNetDeconv.ARCHITECTURE_DICT[self.densenet_name]['deconv_factors'])
        assert(self.deconv_factors.size == len(layers_to_apply_deconv))
        self.outputs_to_concat = list()
        assert num_of_blocks <= len(layers_to_apply_deconv)
        for k in range(num_of_blocks):
            layer_number = layers_to_apply_deconv[k]
            curr_block_output = self.densenet_model.layers[layer_number].output
            self.outputs_to_concat.append(BilinearUpSampling2D(size=(self.deconv_factors[k], self.deconv_factors[k]))(curr_block_output))
        merged_deconv_layers = Concatenate(axis=concat_axis, name='net_feature_map')(self.outputs_to_concat)
        self.backend_model = Model(inputs=self.densenet_model.layers[0].input, outputs=merged_deconv_layers)

    def predict_backend(self, new_sample):
        backend_tensor = self.backend_model.predict(new_sample, 1)
        backend_tensor = np.squeeze(backend_tensor)
        return backend_tensor









