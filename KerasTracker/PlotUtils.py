import matplotlib.pyplot as plt
from Tracker_Params import Tracker_params
import numpy as np
from LDA.utils import get_selection_mask
import UtilFunctions as uf
import json
import os


def plot_lda_masks(image, max_blob_mask, combine_mask, mask_block, mask_color,
                   f1_block, f1_color, frame_number, message, z_score_dict, block_lda_object,  color_lda_object):
    if Tracker_params['is_debug_mode'] and Tracker_params['save_images'] and z_score_dict is not None:
        title_font_size = 24
        fig = plt.figure(figsize=[30, 25])
        plt.subplots_adjust(hspace=0.4)
        num_of_blocks = len(mask_block)

        for k in range(num_of_blocks):
            subplot_var = fig.add_subplot(4, 2, k + 1)
            plt.imshow(mask_block[k], cmap='gray')
            subplot_var.set_title(
                'block %01d Lda - f1: %f' % (k, f1_block[k]), fontsize=title_font_size)

        subplot_var = fig.add_subplot(4,2, 5)
        plt.imshow(mask_color, cmap='gray')
        subplot_var.set_title('color Lda - f1: %f' % f1_color, fontsize=title_font_size)

        subplot_var = fig.add_subplot(4,2, 6)
        plt.imshow(combine_mask, cmap='gray')
        subplot_var.set_title('combined mask', fontsize=title_font_size)

        subplot_var = fig.add_subplot(4,2,7)
        plt.imshow(max_blob_mask[:, :, 0], cmap='gray')
        subplot_var.set_title('max-blob-mask', fontsize=title_font_size)

        subplot_var = fig.add_subplot(4,2,8)
        image[0, :, :, 0] += 123.68
        image[0, :, :, 1] += 116.779
        image[0, :, :, 2] += 103.939
        image = image.astype('uint8')
        img_overlay = uf.create_image_overlaid_with_rotated_rect(image[0, :, :, :], max_blob_mask)
        plt.imshow(img_overlay)
        subplot_var.set_title('image patch with overlaid rect', fontsize=title_font_size)
        debug_path_current = Tracker_params['debug_images_path_temp']
        fig.suptitle(message + json.dumps(z_score_dict), fontsize=title_font_size)
        frame_file_name = 'all_lda_masks_%04d.png' % frame_number
        plt.savefig(os.path.join(debug_path_current, frame_file_name) ,bbox_inches='tight')
        plt.close(fig)


def plot_lda_related(feature_maps, inds_arrays_background, inds_arrays_foreground, image, frame_number, stds, frame_mask, lda_identifier = ''):
    num_of_feature_maps = feature_maps.shape[2]
    image = np.squeeze(image)
    im_size = image.shape[0]
    num_of_rows = 3
    fig = plt.figure(figsize=[20, 15])
    for k in range(num_of_feature_maps):
        subplot_var = fig.add_subplot(num_of_rows, num_of_feature_maps + 1, k + 1)
        imgplot = plt.imshow(feature_maps[:, :, k], cmap='gray')
        imgplot.set_clim(0.0, 0.7)
        subplot_var.set_title('LDA map %02d' % k + 1)
        sublot_var = fig.add_subplot(num_of_rows, num_of_feature_maps + 1, num_of_feature_maps + k + 2)
        selection_mask = get_selection_mask(im_size, inds_arrays_background[k])
        plt.imshow(selection_mask, cmap='gray')
        sublot_var.set_title('Background inds for LDA %02d' % (k + 1))

    final_map = np.amin(feature_maps, axis=-1)
    subplot_var = fig.add_subplot(num_of_rows, num_of_feature_maps + 1, 2 * num_of_feature_maps + 2)
    selection_mask = get_selection_mask(im_size, inds_arrays_foreground)
    plt.imshow(selection_mask, cmap='gray')
    subplot_var.set_title('Foreground inds')

    subplot_var = fig.add_subplot(num_of_rows, num_of_feature_maps + 1, 2 * num_of_feature_maps + 3)
    imgplot = plt.imshow(final_map, cmap='gray')
    imgplot.set_clim(0.0, 0.7)
    subplot_var.set_title('Map of LDA Block')

    subplot_var = fig.add_subplot(num_of_rows, num_of_feature_maps + 1, num_of_feature_maps + 1)
    image[:, :, 0] += 123.68
    image[:, :, 1] += 116.779
    image[:, :, 2] += 103.939
    image = image.astype('uint8')
    plt.imshow(image)
    subplot_var.set_title('The image patch')
    fig.suptitle('debug figure for frame %02d. Mean std: %5.3f. Identifier: %s' % (frame_number, np.mean(stds), lda_identifier), fontsize=16)
    if Tracker_params['save_images']:
        debug_path_current = Tracker_params['debug_images_path_temp']
        plt.savefig("%s/%s_debugImage%04d.png" % (debug_path_current, lda_identifier, frame_number), bbox_inches='tight')
    plt.close(fig)