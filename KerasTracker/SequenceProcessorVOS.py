import matplotlib
matplotlib.use('Agg')
import os
import glob
import shutil
import numpy as np
import UtilFunctions as uf
from Tracker import Tracker
from Tracker_Params import Tracker_params
DEBUG_MODE = Tracker_params['is_debug_mode']


def process_single_sequence(sequence_name, params, result_path, debug_path,  base_vos_path):
    run_data = do_all_preparations(sequence_name, result_path, debug_path, base_vos_path, params)
    num_frames_to_track = len(run_data['image_paths'])

    for m in range(1, num_frames_to_track):
        image_path = run_data['image_paths'][m]
        img_2_read = uf.read_image(image_path)
        predict_mask = single_image_analyze(run_data, img_2_read, m)
        if Tracker_params['save_images']:
            uf.save_mask_image(predict_mask, run_data['result_path'], os.path.basename(image_path))


def prepare_all_paths(sequence_name, result_path, debug_path, base_vos_path, is_hd=False):
    suffix_str = '/1080p/' if is_hd else '/480p/'
    sequence_path_images = base_vos_path + '/JPEGImages' + suffix_str + sequence_name
    sequence_path_masks = base_vos_path +  '/Annotations' + suffix_str + sequence_name
    result_path_sequence = result_path + sequence_name + '/'
    debug_path_sequence = debug_path + sequence_name + '/'
    if os.path.exists(result_path_sequence):
        shutil.rmtree(result_path_sequence)
    os.makedirs(result_path_sequence)
    if DEBUG_MODE:
        if os.path.exists(debug_path_sequence):
            shutil.rmtree(debug_path_sequence)
        os.makedirs(debug_path_sequence)
    return sequence_path_images, sequence_path_masks, result_path_sequence, debug_path_sequence


def get_all_image_paths(sequence_path):
    image_paths = glob.glob(sequence_path + '/*.*')
    image_paths = sorted(image_paths)
    return image_paths


def do_all_preparations(sequence_name, result_path, debug_path, base_vos_path, params):
    images_path, masks_path,  result_path, debug_path = prepare_all_paths(sequence_name, result_path, debug_path, base_vos_path)
    image_paths = get_all_image_paths(images_path)
    mask_paths = get_all_image_paths(masks_path)
    initial_mask = uf.read_image(mask_paths[0])/255
    initial_mask = np.expand_dims(initial_mask, axis=-1)
    if Tracker_params['save_images']:
        uf.save_mask_image(initial_mask, result_path, os.path.basename(image_paths[0]))
    test_tracker = Tracker(uf.read_image(image_paths[0]), None, initial_mask, **params)
    run_params_data = dict({'test_tracker': test_tracker, 'image_paths': image_paths
                               ,'result_path': result_path})
    Tracker_params['debug_images_path_temp'] = debug_path
    return run_params_data


def single_image_analyze(run_data, next_image, image_indx):
    try:
        predict_mask = run_data['test_tracker'].step(next_image, image_indx)
    except Exception as e:
        predict_mask = np.zeros_like(next_image)
    return predict_mask
