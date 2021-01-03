% This script can be used to interactively inspect the results

addpath('D:\Felix\Aharon Bar Hillel Repo\Sandboxes\Felix\vot-toolkit-master'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

trackers = tracker_load('LDATracker');

workspace_browse(trackers, sequences, experiments);

