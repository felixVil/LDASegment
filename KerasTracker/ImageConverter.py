import UtilFunctions as uf
import numpy as np
from skimage.transform import resize
import keras.backend as K



class ImageConverter:

    def __init__(self, input_shape, alpha, original_shape):
        self.alpha = alpha
        self.input_shape = input_shape
        self.original_shape = original_shape

    def create_mask_matrix_for_display(self, mask):
        mask_embedded = uf.embed_mask_into_image(mask, self.crop_rect_for_mask, self.original_shape)
        mask_resized_standard = uf.fix_dimensions(mask_embedded)
        return mask_resized_standard

    def prepare_image_for_prediction(self, img, curr_mask, is_densenet=False,  is_mask=False):
        self.crop_rect_for_mask = uf.create_single_crop_rect(self.alpha, curr_mask,
                                                    img.shape[1], img.shape[0], self.input_shape)
        img_cropped = uf.create_cropped_image(img, self.crop_rect_for_mask)
        im = ImageConverter.create_image_matrix_for_net(img_cropped, self.input_shape, is_densenet, is_mask)
        if is_mask:
            num_of_colours = 1
        else:
            num_of_colours = 3
        im_sample = np.zeros([1, self.input_shape[0], self.input_shape[1], num_of_colours])
        im_sample[0, :, :, :] = im
        return im_sample

    @staticmethod
    def create_image_matrix_for_net(img, input_shape, is_densenet, is_mask=False):
        if is_mask:
            input_shape = (input_shape[0], input_shape[1])
        im_resized = resize(img, input_shape, mode='reflect', preserve_range=True)
        if not is_mask:
            if is_densenet:
                #pre-processing required for DenseNet in Keras.
                x = im_resized / 255
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                if K.image_data_format() == 'channels_first':
                    if x.ndim == 3:
                        x[0, :, :] -= mean[0]
                        x[1, :, :] -= mean[1]
                        x[2, :, :] -= mean[2]
                        if std is not None:
                            x[0, :, :] /= std[0]
                            x[1, :, :] /= std[1]
                            x[2, :, :] /= std[2]
                    else:
                        x[:, 0, :, :] -= mean[0]
                        x[:, 1, :, :] -= mean[1]
                        x[:, 2, :, :] -= mean[2]
                        if std is not None:
                            x[:, 0, :, :] /= std[0]
                            x[:, 1, :, :] /= std[1]
                            x[:, 2, :, :] /= std[2]
                else:
                    x[..., 0] -= mean[0]
                    x[..., 1] -= mean[1]
                    x[..., 2] -= mean[2]
                    if std is not None:
                        x[..., 0] /= std[0]
                        x[..., 1] /= std[1]
                        x[..., 2] /= std[2]
                im_resized = x
            elif np.amax(im_resized) > 128:
                im_resized[:, :, 0] -= 123.68
                im_resized[:, :, 1] -= 116.779
                im_resized[:, :, 2] -= 103.939
        return im_resized



