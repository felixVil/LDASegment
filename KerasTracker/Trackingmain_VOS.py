from Tracker_Params import Tracker_params
import SequenceProcessorVOS as sp
import os
import gc
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#get only validation sequences!!!#
validation_filepath = 'D:/Felix/DAVIS Stuff/DAVIS/ImageSets/480p/val.txt'
with open(validation_filepath) as f:
    content = f.readlines()

sequences = [line.split('/')[3] for line in content]
sequence_names = sorted(list(set(sequences)))

curr_params = Tracker_params
date_str = datetime.datetime.now().strftime("%B_%d_%Y_%H_%M_%S")

descriptor_string = "LDA_%s/" % (date_str)
result_path = curr_params['vos_results_path']
debug_path = curr_params['debug_images_path'] + descriptor_string
base_vos_path = curr_params['base_vos_path']
if not os.path.exists(result_path):
    os.makedirs(result_path)

log_file_path = os.path.join(result_path, 'sequence_run.log')

with open(os.path.join(result_path, 'sequence_run.log'), "w") as log_file:
    for sequence_name in sequence_names:
        try:
            sp.process_single_sequence(sequence_name, curr_params, result_path, debug_path, base_vos_path)
            Tracker_params['debug_images_path_temp'] = ''
        except Exception as e:  # most generic exception you can catch
            log_file.write("Failed to finish sequence %s: %s\n" % (sequence_name, e))
        finally:
            pass

gc.collect()
