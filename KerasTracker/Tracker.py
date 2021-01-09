import UtilFunctions as uf
import numpy as np
from skimage import color
from Trackernets import VGGDeconv, DenseNetDeconv
from LDA.updater import LDAHandler
from ImageConverter import ImageConverter
from math import exp
from PlotUtils import plot_lda_masks
from SegmentationLogic import ZScoreCalculator


class Tracker:

    def __init__(self, initial_img, initial_poly_array, initial_mask, **tracker_args):
        for key, value in tracker_args.items():
            setattr(self, key, value)
        self.feature_map_nums_densenet = DenseNetDeconv.ARCHITECTURE_DICT[self.densenet_name]['feature_maps']
        self.feature_map_nums_vgg19 = VGGDeconv.ARCHITECTURE_DICT['feature_maps']
        if initial_poly_array is not None:
            initial_mask = uf.create_initial_mask(initial_img, initial_poly_array)
        if uf.check_if_small_mask(initial_mask, self.small_mask_ratio):
            print('Small sized object!\n')
            self.input_shape = (160, 160, 3)
            self.alpha = 2.5
        else:
            print('Normal sized object!\n')

        print(self.input_shape)#debug print.

        if self.use_densenet_backend:
            self.deconv_factors = DenseNetDeconv.ARCHITECTURE_DICT['DenseNet121']['deconv_factors']
            self.trackernet = DenseNetDeconv(self.input_shape, self.densenet_name, self.num_of_blocks)
        else:
            self.deconv_factors = VGGDeconv.ARCHITECTURE_DICT['deconv_factors']
            self.trackernet = VGGDeconv(self.feature_map_nums_vgg19, self.input_shape)

        self.image_converter = ImageConverter(self.input_shape, self.alpha, initial_img.shape)
        self.curr_mask = np.copy(initial_mask)
        self.z_score_converter = None
        self.curr_mask_standard = None
        self.tracker_state ='baseline_tracking'
        self.predict_mask_color = None
        self.prev_coeff_color = 0

        self.f1_color = 1
        self.f1_block = [1 for _ in range(self.num_of_blocks)]
        self.predict_mask_block = [None for _ in range(self.num_of_blocks)]

        self.prev_coeff_block = [0 for _ in range(self.num_of_blocks)]

        self.is_first = True
        self.backend_tensor, self.color_tensor = (None for _ in range(2))
        image_2d = (self.input_shape[0], self.input_shape[1])
        self.lda_handler_block = []
        self.prepare_lda_constructors(tracker_args, image_2d)

        self.prepare_inputs(initial_img)
        self.create_standard_mask()
        self.update_all_ldas()
        self.step_ldas()
        self.predict_mask = np.copy(self.curr_mask_standard[:, :, 0])
        self.combined_mask = None
        self.frame_number = 1

    def step(self, new_img, frame_number):
        self.frame_number = frame_number
        self.prepare_inputs(new_img)

        self.obtain_predict_mask_from_ldas()
        self.postprocess_frame()
        self.backend_tensor, self.color_tensor = (None for _ in range(2))
        self.do_plots()

        return self.curr_mask

    def prepare_inputs(self, img):
        self.curr_image_for_prediction_for_color = self.image_converter.prepare_image_for_prediction(img, self.curr_mask)
        self.color_tensor = color.rgb2yuv(self.curr_image_for_prediction_for_color[0, :, :, :])
        self.curr_image_for_prediction = self.image_converter.prepare_image_for_prediction(img, self.curr_mask, self.use_densenet_backend)
        self.backend_tensor = self.trackernet.predict_backend(self.curr_image_for_prediction)

#LDA related

    def do_plots(self):

        if self.z_score_converter is not None:
            z_score_dict = dict(gradient_score=self.z_score_converter.gradient_score,
                                confidence_score = self.z_score_converter.confidence_score,
                                selected_ind=self.z_score_converter.selected_ind,
                                threshold_selected=self.z_score_converter.threshold_selected,
                                volume_nominator=self.z_score_converter.volume_nominator,
                                volume_score=self.z_score_converter.volume_score,
                                displacement=self.z_score_converter.displacement,
                                displacement_score=self.z_score_converter.displacement_score,
                                is_ZOH_used=self.z_score_converter.is_zoh_used,
                                min_z_score=self.z_score_converter.min_z_score)
        else:
            z_score_dict = None


        if self.use_densenet_backend:
            image_patch_display = self.curr_image_for_prediction_for_color
        else:
            image_patch_display = self.curr_image_for_prediction

        plot_lda_masks(np.copy(image_patch_display), self.curr_mask_standard,
                       self.predict_mask, self.predict_mask_block,
                       self.predict_mask_color, self.f1_block,
                       self.f1_color, self.frame_number, self.tracker_state, z_score_dict,
                       self.lda_handler_block,
                       self.lda_handler_color)


        if self.plot_lda_related:
            for k, lda_block in enumerate(self.lda_handler_block):
                lda_block.plot_inds_and_feature_maps(np.copy(image_patch_display), self.predict_mask_block[k], self.frame_number)

            self.lda_handler_color.plot_inds_and_feature_maps(np.copy(image_patch_display), self.predict_mask_color, self.frame_number)

    def obtain_predict_mask_from_ldas(self):
        self.step_ldas()
        self.combine_multi_level_predict_masks()

    def predict_all_ldas(self):
        self.predict_mask_color = self.lda_handler_color.lda_predict_step(self.color_tensor)
        for k in range(len(self.lda_handler_block)):
            self.predict_mask_block[k] = self.lda_handler_block[k].lda_predict_step(self.backend_tensor)

    def update_all_ldas(self):
        if self.f1_color > 0:
            self.lda_handler_color.lda_update_step(self.curr_mask_standard, self.color_tensor)
        for k in range(len(self.lda_handler_block)):
            if self.f1_block[k] > 0:
                self.lda_handler_block[k].lda_update_step(self.curr_mask_standard, self.backend_tensor)

    def step_ldas(self):
        self.predict_all_ldas()

        if self.is_first:
            self.compute_f1s()
            self.is_first = False

    def compute_f1s(self):
        self.f1_color = uf.check_f1(self.curr_mask_standard, self.predict_mask_color, False)
        for k in range(len( self.f1_block)):
            self.f1_block[k] = uf.check_f1(self.curr_mask_standard, self.predict_mask_block[k], False)

    def prepare_lda_constructors(self, tracker_args, image_2d):
        tensor_inds_to_take_color = np.arange(0, 3)
        tensor_inds_to_take_block = [0 for _ in range(self.num_of_blocks)]
        index_sum = 0
        if self.use_densenet_backend:
            for k in range(self.num_of_blocks):
                tensor_inds_to_take_block[k] = np.arange(index_sum, index_sum + self.feature_map_nums_densenet[k])
                index_sum += self.feature_map_nums_densenet[k]

        else:
            for k in range(self.num_of_blocks):
                tensor_inds_to_take_block[k] = np.arange(index_sum, index_sum + self.feature_map_nums_vgg19[k])
                index_sum += self.feature_map_nums_vgg19[k]

        self.lda_handler_color = LDAHandler(tracker_args['num_of_ldas_color'], image_2d, 'color_features_LDA', True,
                                            tensor_inds_to_take_color)
        for k in range(self.num_of_blocks):
            self.lda_handler_block.append(LDAHandler(tracker_args['num_of_ldas'][k], image_2d,
                                                   'block%d_features_LDA' % (k + 1), True,
                                                   tensor_inds_to_take_block[k], self.deconv_factors[k]))

    def combine_multi_level_predict_masks(self):
        self.predict_mask = self.create_color_mask()
        if np.all(self.predict_mask == 0) or uf.check_f1(self.curr_mask_standard, self.predict_mask, True) < self.f1_total_thresh:
            self.predict_mask = self.lda_feature_combine()
            self.tracker_state = 'baseline_tracking'
        else:
            self.tracker_state = 'color_tracking'

    def lda_feature_combine(self):
        f1_color_exp = self.f1_alpha * exp(self.exp_scale_factor * self.f1_color) + \
                       (1 - self.f1_alpha) * self.prev_coeff_color
        self.prev_coeff_color = f1_color_exp
        f1_block_exp = np.zeros(self.num_of_blocks)
        for k in range(self.num_of_blocks):
            f1_block_exp[k] = self.f1_alpha * exp(self.exp_scale_factor * self.f1_block[k]) + \
                              (1 - self.f1_alpha) * self.prev_coeff_block[k]
            self.prev_coeff_block[k] = f1_block_exp[k]
        norm_constant = f1_color_exp + np.sum(f1_block_exp)
        mask_lin_comp = 0
        for k in range(self.num_of_blocks):
            mask_lin_comp += self.predict_mask_block[k] * f1_block_exp[k]
        predict_mask = self.predict_mask_color * f1_color_exp + mask_lin_comp
        predict_mask = predict_mask/norm_constant
        return predict_mask

    def create_color_mask(self):
        if self.f1_color > self.thresh_initial_color:
            predict_mask_based_on_color = self.predict_mask_color
        else:
            predict_mask_based_on_color = 0

        return predict_mask_based_on_color

## Postprocessing related.

    def create_standard_mask(self):
        self.curr_mask_standard = self.create_small_mask(self.curr_mask)

    def create_small_mask(self, big_mask):
        small_mask = self.image_converter.prepare_image_for_prediction(big_mask, big_mask, self.use_densenet_backend, True)
        small_mask = np.around(uf.fix_dimensions(small_mask))
        return small_mask

    def postprocess_frame(self):
        zoh_mask_option = self.create_small_mask(self.curr_mask)
        self.z_score_converter = ZScoreCalculator(self.input_shape[:2], self.curr_mask, zoh_mask_option)

        if self.use_combine_only_in_segmentation:
            #segmentation logic uses only combined map.
            map_tensor = np.expand_dims(self.predict_mask, 0)
        else:
            map_tensor = np.stack((self.predict_mask, self.predict_mask_block[2],
                                self.predict_mask_block[1], self.predict_mask_block[0]))

        z_score_return_values = self.z_score_converter.compute_final_mask(map_tensor, self.image_converter)
        if z_score_return_values[0] is not None:

            self.curr_mask, self.curr_mask_standard = z_score_return_values
            self.compute_f1s()

            if not (self.z_score_converter.is_zoh_used):
                self.update_all_ldas() # don't update LDAs when using zoh  - emergency option
















