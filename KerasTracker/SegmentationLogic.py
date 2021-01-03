from UtilFunctions import *
import numpy as np
from math import pow
from Tracker_Params import Segmentation_params


class ZScoreCalculator:


    def __init__(self, input_shape, curr_mask_initial, zoh_mask_option):
        self.zoh_mask_option = zoh_mask_option
        nonzero_output_zoh = np.nonzero(self.zoh_mask_option)
        self.zoh_coords = np.column_stack([nonzero_output_zoh[0], nonzero_output_zoh[1]])
        self.image_converter = None
        self.volume_score_multiplier = np.square(np.log(1.5))
        self.image_shape = input_shape
        self.cbb_prev = get_mask_center(curr_mask_initial)

        self.vbb_prev_width, self.vbb_prev_height, self.vbb_prev_angle, self.sbb_prev = ZScoreCalculator.compute_vbb_sbb(curr_mask_initial)
        self.gradient_score, self.threshold, self.volume_score, self.volume_nominator, self.displacement,\
        self.variance_ratio, self.displacement_score, self.threshold_selected,\
        self.size_score, self.confidence_score = (None for _ in range(10))
        self.all_region_centers, self.all_regions_curr_thresh = ([] for _ in range(2))

        self.prev_area = np.count_nonzero(curr_mask_initial)
        self.min_z_score = np.inf
        self.min_gradient_score, self.min_confidence_score = (np.inf for _ in range(2))
        self.selected_region_coords = None
        self.is_zoh_used = False
        self.selected_ind = None
        self.mask_map_tensor = None

        #switches to use in ablation study to turn"off" different loss elements. Use 1 in normal operation.
        self.enable_grad = float(Segmentation_params['enable_grad'])
        self.enable_confidence = float(Segmentation_params['enable_confidence'])
        self.enable_volume = float(Segmentation_params['enable_volume'])
        self.enable_displacement = float(Segmentation_params['enable_displacement'])

    def compute_z_score_elements(self, blob_mask_image, map_image):

        big_blob_mask_image = self.compute_bigger_mask(blob_mask_image)
        cbb = get_mask_center(big_blob_mask_image)

        vbb_width, vbb_height, vbb_angle, sbb = ZScoreCalculator.compute_vbb_sbb(big_blob_mask_image)
        self.compute_displacement_and_volume_scores(cbb, vbb_width, vbb_height)
        gradient_avg = ZScoreCalculator.compute_average_gradient(blob_mask_image, map_image)
        core_gradient_score = float(1 - Segmentation_params['alpha_grad'] * gradient_avg)
        self.gradient_score = Segmentation_params['grad_score_coeff'] * \
                              pow(np.maximum(core_gradient_score, 0),
                                  Segmentation_params['power_gradient_score'])

        self.confidence_score = 1 - ZScoreCalculator.get_blob_confidence(self.mask_map_tensor, blob_mask_image)
        z_score = self.gradient_score * self.enable_grad + self.volume_score * self.enable_volume\
                  + self.displacement_score * self.enable_displacement + self.confidence_score * self.enable_confidence
        return z_score, sbb, vbb_width, vbb_height, vbb_angle, cbb

    def create_mask_from_coords(self, coords):
        mask = np.zeros(self.image_shape, dtype='float64')
        mask[coords[:, 0], coords[:, 1]] = 1
        return mask

    def compute_score_from_coords(self, coords, map_image):
        blob_mask_image = self.create_mask_from_coords(coords)
        z_score, sbb, vbb_width, vbb_height, vbb_angle, cbb = self.compute_z_score_elements(blob_mask_image, map_image)
        return z_score, sbb, vbb_width, vbb_height, vbb_angle, cbb

    def compute_final_mask(self, mask_map_tensor, image_converter):
        self.mask_map_tensor = mask_map_tensor
        self.image_converter = image_converter
        self.search_for_best_score(mask_map_tensor)
        final_mask = self.create_mask_from_coords(self.selected_region_coords)
        selected_mask = fix_dimensions(final_mask)

        #needed  to compute correct score stats for plotting
        self.compute_score_from_coords(self.selected_region_coords, mask_map_tensor[self.selected_ind, :, :])
        big_final_mask = self.compute_bigger_mask(final_mask)
        return big_final_mask, selected_mask

    def search_for_best_score(self, mask_map_tensor):
        for k in range(mask_map_tensor.shape[0]):
            self.process_single_feature_map(mask_map_tensor, k)
        self.prev_area = self.compute_big_mask_area(self.selected_region_coords)

    def process_single_feature_map(self, mask_map_tensor, feature_ind):
        mask_map_image = norm_mask_by_max(mask_map_tensor[feature_ind, :, :])

        has_been_updated_zoh = self.compute_update_score(self.zoh_coords, mask_map_image, -1)

        if has_been_updated_zoh:
            self.selected_region_coords = np.copy(self.zoh_coords)
            self.is_zoh_used = True

        for self.threshold in Segmentation_params['thresholds']:
            self.get_all_regions_data(mask_map_image, self.threshold)
            for region in self.all_regions_curr_thresh:
                self.try_find_anchor(region, mask_map_image, feature_ind)

    def try_find_anchor(self, region, mask_map_image, feature_ind):
        selected_region_center = None
        has_been_updated = False
        if Segmentation_params['anchor_threshold_range'][1] >= self.threshold >= Segmentation_params['anchor_threshold_range'][0]:
            has_been_updated = self.compute_update_score(region.coords, mask_map_image, feature_ind)
            if has_been_updated:
                self.is_zoh_used = False
                selected_region_center = np.copy(region.centroid)
                self.selected_region_coords = np.copy(region.coords)


        return selected_region_center, has_been_updated

    def get_all_regions_data(self, mask_map_image, threshold):
        all_regions, all_region_centers = ([] for _ in range(2))
        regions = get_regions_from_mask(mask_map_image, threshold=threshold)
        for region in regions:
            region_mask_big_area = self.compute_big_mask_area(region.coords)
            if self.prev_area * Segmentation_params['min_area_mult'] < region_mask_big_area < \
                    self.prev_area * Segmentation_params['max_area_mult']:
                all_regions.append(region)
                all_region_centers.append(region.centroid)
        self.all_regions_curr_thresh = all_regions
        self.all_region_centers = all_region_centers

    def compute_big_mask_area(self, coords):
        region_mask_big = self.compute_bigger_mask(self.create_mask_from_coords(coords))
        region_mask_big_area = np.count_nonzero(region_mask_big)
        return region_mask_big_area

    def compute_bigger_mask(self, blob_mask_image):
        blob_mask_bigger = self.image_converter.create_mask_matrix_for_display(blob_mask_image)
        blob_mask_bigger = np.around(blob_mask_bigger)
        return blob_mask_bigger

    def compute_displacement_and_volume_scores(self, cbb, vbb_width, vbb_height):
        self.volume_nominator = np.square(np.log(vbb_width * vbb_height / (self.vbb_prev_width * self.vbb_prev_height)))
        volume_score_new = self.volume_nominator / self.volume_score_multiplier

        self.volume_score = Segmentation_params['volume_score_coeff'] * volume_score_new
        self.displacement = np.linalg.norm(cbb - self.cbb_prev)

        displacement_score_new = self.displacement / (self.sbb_prev + np.finfo(float).eps)
        self.displacement_score = displacement_score_new * Segmentation_params['displacement_score_coeff']

    def compute_update_score(self, coords,  mask_map_image, feature_ind):
        has_been_updated_now = False
        z_score, sbb, vbb_width,\
        vbb_height, vbb_angle, cbb = self.compute_score_from_coords(coords, mask_map_image)

        if z_score < self.min_z_score:
            self.min_confidence_score = self.confidence_score
            self.min_gradient_score = self.gradient_score
            self.selected_ind = feature_ind
            self.min_z_score = z_score
            has_been_updated_now = True
            self.threshold_selected = self.threshold

        return has_been_updated_now


    @staticmethod
    def compute_variance_from_inds(coords, map_image):
        values = []
        for k in range(coords.shape[0]):
            values.append(map_image[coords[k, 0], coords[k, 1]])
        values = np.asarray(values)
        return np.var(values)

    @staticmethod
    def compute_distances_from_center(center, points):
        distances = []
        for point in points:
            distances.append(np.linalg.norm(np.array([center[0] - point[0], center[1] - point[1]])))
        distances = np.array(distances)
        indices = np.argsort(distances)
        return distances, indices

    @staticmethod
    def compute_vbb_sbb(blob_mask_image):
        if np.amax(blob_mask_image) > 0:
            _, _, rect = convert_mask_to_rotated_rect(blob_mask_image)
            bb_width = rect[1][0]
            bb_height = rect[1][1]
            bb_angle = rect[2]
            sbb = np.linalg.norm(np.array([bb_width, bb_height]))
        else:
            bb_width, bb_height, bb_angle, sbb = (0 for _ in range(4))
        return bb_width, bb_height, bb_angle, sbb

    @staticmethod
    def compute_complementary_mask_inds(mask):
        label_mask = label(1 - mask)
        regions = measure.regionprops(label_mask)
        if regions:
            complementary_coords = regions[0].coords
        else:
            complementary_coords = np.array([])
        return complementary_coords

    @staticmethod
    def compute_average_gradient(mask_image, predict_mask):
        gradient_sum_top, num_of_points_top = ZScoreCalculator.get_gradient_vector(mask_image, predict_mask, direction='top')
        gradient_sum_bottom, num_of_points_bottom = ZScoreCalculator.get_gradient_vector(mask_image, predict_mask, direction='bottom')
        gradient_sum_left, num_of_points_left = ZScoreCalculator.get_gradient_vector(mask_image, predict_mask, direction='left')
        gradient_sum_right, num_of_points_right = ZScoreCalculator.get_gradient_vector(mask_image, predict_mask, direction='right')
        gradient_sum = gradient_sum_top + gradient_sum_bottom + gradient_sum_left + gradient_sum_right
        total_num_of_points = num_of_points_top + num_of_points_bottom + num_of_points_left + num_of_points_right
        total_num_of_points = max(total_num_of_points, 1)  # avoid division by zero.
        return gradient_sum / total_num_of_points

    @staticmethod
    def get_gradient_vector(mask_image, predict_mask, direction='top'):
        dx, dy = ZScoreCalculator.get_dx_dy_for_grad_vec(direction)

        if dy == -1 and dx == 0:
            y_inds, x_inds = np.nonzero(mask_image[0:dy, dx:] - mask_image[-dy:, :] > 0)
            y_inds += 1
        elif dy == 1 and dx == 0:
            y_inds, x_inds = np.nonzero(mask_image[dy:, dx:] - mask_image[0:-dy, :] > 0)
        elif dy == 0 and dx == -1:
            y_inds, x_inds = np.nonzero(mask_image[dy:, 0:dx] - mask_image[:, -dx:] > 0)
            x_inds += 1
        elif dy == 0 and dx == 1:
            y_inds, x_inds = np.nonzero(mask_image[dy:, dx:] - mask_image[:, 0:-dx] > 0)

        num_of_points = x_inds.size
        gradient_sum = 0

        for k in range(x_inds.size):
            gradient_sum += predict_mask[y_inds[k] + dy, x_inds[k] + dx] - predict_mask[y_inds[k], x_inds[k]]

        return gradient_sum, num_of_points

    @staticmethod
    def get_dx_dy_for_grad_vec(direction):
        if direction == 'top':
            dx = 0
            dy = 1
        elif direction == 'bottom':
            dx = 0
            dy = -1
        elif direction == 'left':
            dx = 1
            dy = 0
        elif direction == 'right':
            dx = -1
            dy = 0
        else:
            raise ValueError('direction should be one of: top, bottom, left, right')
        return dx, dy

    @staticmethod
    def recenter_coords(coords, new_center, shape_tuple):
        new_center_array = np.asarray(new_center)
        coords_center = np.mean(coords, axis=0)
        new_coords = coords - coords_center + new_center_array
        new_coords = np.asarray(np.around(new_coords), np.int32)
        log_indices_to_take = np.logical_and(
            np.logical_and(new_coords[:, 0] < shape_tuple[0], new_coords[:, 1] < shape_tuple[1]),
            np.logical_and(0 <= new_coords[:, 1], 0 <= new_coords[:, 0]))
        indices_to_take = np.nonzero(log_indices_to_take)
        final_coords = np.take(new_coords, indices_to_take[0], axis=0)
        return final_coords

    @staticmethod
    def get_blob_confidence(mask_map_tensor, blob_mask_image, num_of_max_points=10, ref_ind=0):
        blobed_mask_map = mask_map_tensor[ref_ind, :, :] * blob_mask_image
        decending_order_sorted = np.sort(blobed_mask_map, axis=None)[::-1]
        confidence_value = np.mean(decending_order_sorted[:num_of_max_points])
        assert 0 <= confidence_value <= 1
        return confidence_value



