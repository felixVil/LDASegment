from LDA.utils import *
import numpy as np
from LDA.lda_calc import LDANormalized
from PlotUtils import plot_lda_related


class LDAHandler:


    def __init__(self, num_of_lda_features, imshape_2d, identifier, do_regularize_matrix, tensor_inds_to_take, deconv_factor=1):
        self.identifier = identifier
        self.deconv_factor = deconv_factor
        self.do_regularize_matrix = do_regularize_matrix
        self.num_of_LDA_backgrounds = num_of_lda_features
        self.imshape_2d = imshape_2d
        self.tensor_inds_to_take = tensor_inds_to_take
        self.sklearn_ldas = [LDANormalized(solver='lsqr') for _ in range(self.num_of_LDA_backgrounds)]
        self.feature_maps = None
        self.foreground_vectors = None
        self.foreground_mask_2d_inds = None
        self.background_inds_groups = None
        self.background_vectors = [None for _ in range(self.num_of_LDA_backgrounds)]
        self.covariances, self.mean_vecs = ([None for _ in range(self.num_of_LDA_backgrounds)] for _ in range(2))
        self.foreground_covariance, self.foreground_mean = (None for _ in range(2))
        self.background_covariance, self.background_mean = ([None for _ in range(self.num_of_LDA_backgrounds)] for _ in range(2))
        self.num_of_vecs_in_background = np.array([0 for _ in range(self.num_of_LDA_backgrounds)])
        self.num_of_vecs_in_foreground = 0
        self.priors = np.array([0, 0])
        self.stds =[]

    def lda_predict_step(self, feature_map_tensor):
        feature_map_tensor = get_sub_tensor(feature_map_tensor, self.tensor_inds_to_take)
        final_map = self.predict_probs(feature_map_tensor)
        return final_map

    def lda_update_step(self, selection_mask, feature_map_tensor):
        feature_map_tensor = get_sub_tensor(feature_map_tensor, self.tensor_inds_to_take)
        self.prepare_all_lda_vectors_hard(selection_mask, feature_map_tensor)
        self.compute_weight_multi_lda()

    def predict_probs(self, feature_map_tensor):
        # assuming square image.
        output_maps = np.zeros((self.imshape_2d[0], self.imshape_2d[1], self.num_of_LDA_backgrounds))
        feature_vecs = stack_feature_vecs(feature_map_tensor, self.imshape_2d)
        for m in range(self.num_of_LDA_backgrounds):
            output_probs = self.sklearn_ldas[m].predict_proba_limited(feature_vecs)
            for i in range(self.imshape_2d[0]):
                for j in range(self.imshape_2d[1]):
                    output_maps[i, j, m] = output_probs[i * self.imshape_2d[1] + j, -1]
        final_output = np.amin(output_maps, axis=-1)
        stds = np.array([lda.prob_std for lda in self.sklearn_ldas])
        self.stds.append(stds)
        self.feature_maps = output_maps
        return final_output

    def update_means_and_cov_foreground_lda(self):
        num_of_foreground_vecs = self.foreground_vectors.shape[0]
        self.foreground_covariance = compute_accumulated_covariance(self.foreground_vectors,
                                                                    self.num_of_vecs_in_foreground,
                                                                    self.foreground_covariance)
        self.foreground_mean = compute_accumulated_mean(self.foreground_vectors, self.num_of_vecs_in_foreground,
                                                        self.foreground_mean)
        self.num_of_vecs_in_foreground += num_of_foreground_vecs

    def compute_weight_multi_lda(self):
        self.update_means_and_cov_foreground_lda()
        self.foreground_vectors = None
        for m in range(self.num_of_LDA_backgrounds):
            if self.background_vectors[m].size > 0:
                self.compute_lda_weights(m)
        self.background_vectors = [None for _ in range(self.num_of_LDA_backgrounds)]

    def compute_lda_weights(self, background_lda_ind):
        self.update_means_and_covs(background_lda_ind)
        self.sklearn_ldas[background_lda_ind].update_lda_params(self.covariances[background_lda_ind], self.mean_vecs[background_lda_ind], self.priors)

    def extract_foreground_background_inds_hard(self, selection_mask):
        self.foreground_mask_2d_inds, background_mask_inds = get_foreground_and_background_inds(selection_mask, self.deconv_factor)
        self.background_inds_groups = split_background_inds_2_parts(self.foreground_mask_2d_inds, background_mask_inds,
                                                                    self.num_of_LDA_backgrounds)

    def extract_foreground_background_vectors(self, feature_map_tensor):
        for group_ind in range(self.num_of_LDA_backgrounds):
                self.background_vectors[group_ind] = create_lda_feature_vector(feature_map_tensor, self.background_inds_groups[group_ind])
        self.foreground_vectors = create_lda_feature_vector(feature_map_tensor, self.foreground_mask_2d_inds)

    def prepare_all_lda_vectors_hard(self, selection_mask, feature_map_tensor):
        self.extract_foreground_background_inds_hard(selection_mask)
        self.extract_foreground_background_vectors(feature_map_tensor)

    def plot_inds_and_feature_maps(self, image, final_mask_for_frame, frame_number):
        if (self.feature_maps is not None) and (self.background_inds_groups is not None) \
                and (self.foreground_mask_2d_inds is not None) and DEBUG_MODE:
            plot_lda_related(self.feature_maps, self.background_inds_groups, self.foreground_mask_2d_inds,
                             image, frame_number, self.stds[frame_number - 2], final_mask_for_frame, self.identifier)

    def update_means_and_cov_background_lda(self, background_lda_ind):
        background_vectors = self.background_vectors[background_lda_ind]
        self.background_covariance[background_lda_ind] = \
            compute_accumulated_covariance(background_vectors, self.num_of_vecs_in_background[background_lda_ind],
                                           self.background_covariance[background_lda_ind])
        self.background_mean[background_lda_ind] = compute_accumulated_mean(background_vectors,
                                                                            self.num_of_vecs_in_background[
                                                                                background_lda_ind],
                                                                            self.background_mean[background_lda_ind])

    def update_means_and_covs(self, background_lda_ind):
        self.update_means_and_cov_background_lda(background_lda_ind)
        num_of_background_vecs = self.background_vectors[background_lda_ind].shape[0]
        self.num_of_vecs_in_background[background_lda_ind] += num_of_background_vecs
        total_num_vecs = self.num_of_vecs_in_background[background_lda_ind] + self.num_of_vecs_in_foreground
        self.priors = np.array([self.num_of_vecs_in_background[background_lda_ind]/total_num_vecs, self.num_of_vecs_in_foreground/total_num_vecs])
        covariance = self.foreground_covariance * self.priors[1] + self.background_covariance[background_lda_ind] * self.priors[0]
        mean_vecs = np.zeros((2, self.foreground_mean.size))
        mean_vecs[0, :] = self.background_mean[background_lda_ind]
        mean_vecs[1, :] = self.foreground_mean
        if self.do_regularize_matrix:
            covariance = regularize_matrix(covariance)
        self.covariances[background_lda_ind] = covariance
        self.mean_vecs[background_lda_ind] = mean_vecs

