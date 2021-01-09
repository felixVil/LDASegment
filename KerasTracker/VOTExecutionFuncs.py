import json
import os
import time
import numpy as np
from Tracker import Tracker
import UtilFunctions as uf
from Tracker_Params import Tracker_params
import datetime
import os


def initialize(image_file_path, region_json_file):

    date_str = datetime.datetime.now().strftime("%B_%d_%Y_EAO_run_sep")
    sequence_name = image_file_path.split('\\')[-3]
    descriptor_string = "LDA_%s" % (date_str)
    debug_path = os.path.join(Tracker_params['debug_images_path'], descriptor_string, sequence_name)
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
    Tracker_params['debug_images_path_temp'] = debug_path
    print('\nDebug Images Path:' + debug_path + '\n')

    print(image_file_path) #debug print!!!
    with open(region_json_file, 'r') as fid:
        region = json.load(fid)
    print(region)#debug print!!!
    poly_array = np.array(region)
    if poly_array.size < 5: #case of rect
        poly_array = uf.convert_rect_to_real_poly(poly_array)
    tracker_object = Tracker(uf.read_image(image_file_path), poly_array, None, **Tracker_params)
    is_end = False
    script_path = os.path.realpath(__file__)
    info_file_path = os.path.join(os.path.dirname(script_path), 'info.txt')
    while not is_end:
        while (not os.path.isfile(info_file_path)) or os.path.getsize(info_file_path) == 0:
            time.sleep(1)
        if os.path.isfile(info_file_path):
            with open(info_file_path, "r") as f:
                data = f.read()
        if "EndOfSequence" in data:
            is_end = True
        else:
            prev_image_file_path = image_file_path
            image_file_path = data
            print(image_file_path)
            if image_file_path == prev_image_file_path:
                time.sleep(2)
                continue
            frame_number_string = os.path.basename(image_file_path).split('.')[0]
            frame_number = int(frame_number_string)
            new_image = uf.read_image(image_file_path)
            mask_img = tracker_object.step(new_image,frame_number)
            region_rect = uf.get_region_dict_from_mask(mask_img)
            if _is_on_all_image_masking(region_rect, mask_img):
                region = _handle_all_image_masking()
            else:
                region = _get_rotated_rect_from_mask(mask_img)
            json_file_4_update = region_json_file.split('.')[0] + 'Update' + '.json'
            print('update json file is:')
            print(json_file_4_update)#debug print!!!
            with open(json_file_4_update, 'w') as fid:
                json.dump(region, fid, sort_keys=True, indent=2, separators=(',', ': '))


def _get_rotated_rect_from_mask(mask_img):
    region, _, _ = uf.convert_mask_to_rotated_rect(mask_img)
    region_str= list()
    for el in region:
        region_str.append(str(el))
    region = {'region': region_str}
    print(region)  # debug print!!!
    return region


def _is_on_all_image_masking(region_rect, mask_image):
    if region_rect['width'] > 0.97 * mask_image.shape[1] and region_rect['height'] > 0.97 * mask_image.shape[0]:
        return True
    else:
        return False


def _handle_all_image_masking():
    # dummy rect at image corner to simulate failure in case of all image masking.
    region_at_corner = {'region': ['1.0', '1.0', '1.0', '2.0', '2.0', '1.0', '2.0', '2.0']}
    print('\n Handling all image masking!!!!! \n')
    return region_at_corner


