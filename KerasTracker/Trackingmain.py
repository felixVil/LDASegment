from Tracker_Params import Tracker_params
import SequenceProcessor as sp
import os
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sequence_names = ['matrix', 'ball1', 'car1', 'ants3', 'gymnastics1', 'fish1', 'butterfly']

curr_params = Tracker_params
date_str = datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")

descriptor_string = "LDA_%s/" % (date_str)
result_path = curr_params['track_results_path'] + descriptor_string
debug_path = curr_params['debug_images_path'] + descriptor_string
base_vot_path = curr_params['base_vot_path']
os.makedirs(result_path)


for sequence_name in sequence_names:
    sp.process_single_sequence(sequence_name, curr_params, result_path, debug_path, base_vot_path)
    Tracker_params['debug_images_path_temp'] = ''


