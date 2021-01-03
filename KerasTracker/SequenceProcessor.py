import matplotlib
matplotlib.use('Agg')
from skimage.io import imsave
from numpy import genfromtxt
import os
import shutil
import json
import UtilFunctions as uf
from Tracker import Tracker
from Tracker_Params import Tracker_params
DEBUG_MODE = Tracker_params['is_debug_mode']


def process_single_sequence(sequence_name, params, result_path, debug_path,  base_vot_path):
    run_data = do_all_preparations(sequence_name, result_path, debug_path, base_vot_path, params)
    num_frames_to_track = len(run_data['image_paths'])

    for m in range(1, num_frames_to_track):
        image_path = run_data['image_paths'][m]
        img_2_read = uf.read_image(image_path)
        predict_mask, region = single_image_analyze(run_data, img_2_read, m)
        if Tracker_params['write_rect_json']:
            json_file = os.path.basename(image_path).split('.')[0] + '.json'
            json_file_path = os.path.join(run_data['result_path'], json_file)
            with open(json_file_path, 'w') as fid:
                json.dump(region, fid, sort_keys=True, indent=2, separators=(',', ': '))
        if Tracker_params['save_images']:
            uf.save_overlaid_and_mask_images(m, img_2_read, predict_mask, run_data['result_path'])



def prepare_all_paths(sequence_name, result_path, debug_path, base_vot_path):
    sequence_path_sequence = base_vot_path + sequence_name + '/'
    result_path_sequence = result_path + sequence_name + '/'
    debug_path_sequence = debug_path + sequence_name + '/'
    if os.path.exists(result_path_sequence):
        shutil.rmtree(result_path_sequence)
    os.makedirs(result_path_sequence)
    if DEBUG_MODE:
        if os.path.exists(debug_path_sequence):
            shutil.rmtree(debug_path_sequence)
        os.makedirs(debug_path_sequence)
    return sequence_path_sequence, result_path_sequence, debug_path_sequence


def prepare_poly_matrix(sequence_path):
    csv_path = os.path.join(sequence_path, 'groundtruth.txt')
    polygon_matrix = genfromtxt(csv_path, delimiter=',')
    return polygon_matrix


def get_all_image_paths(sequence_path):
    num_of_files = 0
    for file in os.listdir(sequence_path):
        if file.endswith(".jpg"):
            num_of_files = num_of_files + 1
    image_paths = list()
    for k in range(num_of_files):
        filename = "%08d" % (k + 1) + ".jpg"
        image_paths.append(os.path.join(sequence_path, filename))
    return image_paths


def handle_first_image(image_paths, result_path, poly_array):
    img_2_read = uf.read_image(image_paths[0])
    img_overlay = uf.create_image_overlaid_with_polygon(img_2_read, poly_array)
    result_img_filename = 'res_00001.png'
    if Tracker_params['save_images']:
        imsave(os.path.join(result_path, result_img_filename), img_overlay)


def do_all_preparations(sequence_name, result_path, debug_path, base_vot_path, params):
    sequence_path, result_path, debug_path = prepare_all_paths(sequence_name, result_path, debug_path, base_vot_path)
    polygon_matrix = prepare_poly_matrix(sequence_path)
    poly_array = polygon_matrix[0, :]
    image_paths = get_all_image_paths(sequence_path + 'color' + '/')
    handle_first_image(image_paths, result_path, poly_array)
    test_tracker = Tracker(uf.read_image(image_paths[0]), poly_array, **params)
    run_params_data = dict({'test_tracker': test_tracker, 'polygon_matrix': polygon_matrix, 'image_paths': image_paths
                               ,'result_path': result_path})
    Tracker_params['debug_images_path_temp'] = debug_path
    return run_params_data


def single_image_analyze(run_data, next_image, image_indx):
    predict_mask = run_data['test_tracker'].step(next_image, image_indx)
    region = uf.get_region_dict_from_mask(predict_mask)
    return predict_mask, region
