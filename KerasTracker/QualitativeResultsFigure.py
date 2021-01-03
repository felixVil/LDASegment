from UtilFunctions import *
import os
import numpy as np


def find_result_per_sequence_tracker_ind(sequence, tracker, ind):
    result_sequence_path = os.path.join(results_path, tracker, 'baseline')
    result_filename = '%s_001.txt' % sequence
    result_filepath = os.path.join(result_sequence_path, sequence, result_filename)
    file_id = open(result_filepath, 'r')
    lines = file_id.readlines()
    file_id.close()
    polygon_line = lines[ind]
    polygon_line.replace('\n','')
    polygon_arr = np.array([float(element) for element in polygon_line.split(',')])
    return polygon_arr


sequence_path = 'D:/Another_D/E_backup/my homework/BGU Computer Vision thesis/vot-toolkit-master-2019/vot-workspace/sequences'
results_path = 'D:/Another_D/E_backup/my homework/BGU Computer Vision thesis/results_on_tracker_qualitatively_evaluated'
overlay_images_path = 'overlay_images'

if not os.path.exists(overlay_images_path):
    os.makedirs(overlay_images_path)
sequences_dict = {'zebrafish1': {'inds' : [14, 31, 57], 'width': 2},
                  'fish1': {'inds': [143, 278, 316], 'width': 2},
                  'gymnastics2': {'inds': [178, 194, 206], 'width': 9},
                  'book': {'inds': [43, 82, 104], 'width': 2},
                  'conduction1':{'inds': [42, 187], 'width': 2},
                  'dinosaur': {'inds': [220, 277], 'width' : 9}}
color_dict = {'SiamMask':(255, 255, 255, 128), 'UPDT': (255, 0, 255, 128), 'ATOM':(255, 0, 0, 128), 'LADCF': (0, 0, 255, 128), 'LDATrackerDenseNetDilate':(0, 255, 0, 128)}


for sequence in sequences_dict.keys():
    line_width = sequences_dict[sequence]['width']
    poi_inds = sequences_dict[sequence]['inds']
    frames_folder = os.path.join(sequence_path, sequence, 'color')
    for ind in poi_inds:
        poly_arrays = []
        frames_file = os.path.join(frames_folder, '%08d.jpg' % (ind + 1))
        overlay_image_file = os.path.join(overlay_images_path, '%s_%08d.jpg' % (sequence, ind + 1))
        for tracker in color_dict.keys():
            poly_array = find_result_per_sequence_tracker_ind(sequence, tracker, ind)
            if len(poly_array) < 4:
                continue  # tracker is during failure.
            elif len(poly_array) == 4:
                #polygon is a standard axis aligned rectangle.
                poly_array = convert_rect_to_real_poly(poly_array)
            poly_arrays.append(poly_array)
            draw_beatiful_polygon(poly_array, frames_file, overlay_image_file, color_dict[tracker], line_width)
            frames_file = overlay_image_file

        img_overlay = read_image(overlay_image_file)
        crop_rect = create_tight_rect_around_locations(poly_arrays, img_overlay.shape)
        img_overlay_cropped = img_overlay[crop_rect[0]:crop_rect[1], crop_rect[2]:crop_rect[3]]
        img_overlay_cropped_pil = Image.fromarray(img_overlay_cropped, 'RGB')

        overlay_cropped_filename = 'cropped_%s_%08d.png' % (sequence, ind + 1)
        overlay_cropped_filepath = os.path.join(overlay_images_path, overlay_cropped_filename)
        img_overlay_cropped_pil.save(overlay_cropped_filepath, "PNG")


