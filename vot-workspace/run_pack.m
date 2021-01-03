% This script can be used to pack the results and submit them to a challenge.

addpath('D:\Felix\Aharon Bar Hillel Repo\Sandboxes\Felix\vot-toolkit-master'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

%modification removing experiments
experiments_to_use = experiments(1);

%modification removing sequences
sequences_to_use = sequences(1);

tracker = tracker_load('LDATracker');

workspace_submit(tracker, sequences_to_use, experiments_to_use);

