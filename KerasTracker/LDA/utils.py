import numpy as np
import UtilFunctions as uf
from math import pi
import cv2
from Tracker_Params import Tracker_params

STABILITY_EPSILON = Tracker_params['regularization_epsilon']
DEBUG_MODE = Tracker_params['is_debug_mode']


def split_background_inds_2_parts(foreground_mask_2d_inds, background_mask_inds, num_of_parts):
    foreground_center_mass = np.mean(foreground_mask_2d_inds, axis=0)
    background_inds_split = []
    if num_of_parts > 1:
        background_mask_inds_corrected = background_mask_inds - foreground_center_mass
        y_s = background_mask_inds_corrected[:, 0]
        x_s = background_mask_inds_corrected[:, 1]
        polar_coords = [cart2pol(x, y) for x, y in zip(x_s, y_s)]
        inds_angles = [coords[1] for coords in polar_coords]
        bin_vector = np.arange(pi, -pi, -2*pi/num_of_parts)
        group_inds = np.digitize(inds_angles, bin_vector)
        for ind in range(1, bin_vector.size + 1):
            background_inds_split.append(background_mask_inds[group_inds == ind, :])
    else:
        background_inds_split.append(background_mask_inds)
    return background_inds_split


def create_lda_feature_vector(feature_map, mask_2d_inds):
    num_of_2d_inds = mask_2d_inds.shape[0]
    lda_vectors = np.array([feature_map[mask_2d_inds[k, 0], mask_2d_inds[k, 1]] for k in range(num_of_2d_inds)])
    return lda_vectors


def stack_feature_vecs(feature_map_tensor, im_size):
    feature_vecs_stacked = []
    for i in range(im_size[0]):
        for j in range(im_size[1]):
            feature_vecs_stacked.append(feature_map_tensor[i, j, :])
    feature_vecs_stacked = np.vstack(feature_vecs_stacked)
    return feature_vecs_stacked


def get_foreground_and_background_inds(selection_mask, factor):
    foreground_mask_inds = np.nonzero(selection_mask)
    foreground_mask_inds = np.vstack((foreground_mask_inds[0], foreground_mask_inds[1])).T
    background_mask = 1 - selection_mask
    if factor > 1:
        kernel = np.ones((factor - 1, factor - 1))
        background_mask = cv2.erode(background_mask, kernel, iterations=1)
    background_mask_inds = np.nonzero(background_mask)
    background_mask_inds = np.vstack((background_mask_inds[0], background_mask_inds[1])).T
    return foreground_mask_inds, background_mask_inds


def get_selection_mask(im_size, inds_array):
    selection_mask = np.zeros((im_size, im_size))
    num_of_inds = inds_array.shape[0]
    for k in range(num_of_inds):
        selection_mask[inds_array[k, 0], inds_array[k, 1]] = 1
    return selection_mask


def regularize_matrix(cov_matrix):
    dimension_cov = cov_matrix.shape[0]
    regularized_cov_matrix = (1 - STABILITY_EPSILON * dimension_cov) * cov_matrix + np.trace(cov_matrix) * STABILITY_EPSILON * \
    np.eye(dimension_cov)
    return regularized_cov_matrix


def get_sub_tensor(tensor, inds_to_take):
    num_of_inds = len(inds_to_take)
    sliced_tensor = np.zeros((tensor.shape[0], tensor.shape[1], num_of_inds))
    for k in range(num_of_inds):
        sliced_tensor[:, :, k] = tensor[:, :, inds_to_take[k]]
    return sliced_tensor


def compute_accumulated_covariance(X, prev_number_of_vectors= 0, prev_covariance=None):
    num_of_vectors = X.shape[0]
    curr_covariance = np.cov(X.T, bias=True)
    if prev_covariance is None:
        return curr_covariance
    else:
        total_covariance = (curr_covariance * num_of_vectors + prev_covariance * prev_number_of_vectors)/(prev_number_of_vectors + num_of_vectors)
        return total_covariance


def compute_accumulated_mean(X, prev_number_of_vectors= 0, prev_mean=None):
    num_of_vectors = X.shape[0]
    curr_mean = np.mean(X, axis=0)
    if prev_mean is None:
        return curr_mean
    else:
        total_mean = (curr_mean * num_of_vectors + prev_mean * prev_number_of_vectors)/(prev_number_of_vectors + num_of_vectors)
        return total_mean

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)