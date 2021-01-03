% This script can be used to execute the experiments for a single tracker
% You can copy and modify it to create another experiment launcher

addpath('D:\Felix\Aharon Bar Hillel Repo\Sandboxes\Felix\vot-toolkit-master'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

%modification removing experiments
experiments_to_use = experiments(1);

%modification removing sequences
inds = 1:60;

sequences_to_use = sequences(inds);

tracker = tracker_load('LDATracker');

tracker.metadata.deterministic = true;

workspace_evaluate(tracker, sequences_to_use, experiments_to_use);

