% This script can be used to perform a comparative analyis of the experiments
% in the same manner as for the VOT challenge
% You can copy and modify it to create a different analyis

addpath('D:\Felix\Aharon Bar Hillel Repo\Sandboxes\Felix\vot-toolkit-master'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

%error('Analysis not configured! Please edit run_analysis.m file.'); % Remove this line after proper configuration

trackers = tracker_list('LDATracker'); % TODO: add more trackers here

%modification removing experiments
experiments_to_use = experiments(1);

%modification removing sequences
%inds = [4, 14, 18, 21, 22, 33, 34, 36, 42, 48, 53, 56];
inds = 1:60; 
sequences_to_use = sequences(inds);


workspace_analyze(trackers, sequences_to_use, experiments_to_use, 'report_LDATracker_AbelationFullNoGradient', 'Title', 'Report for vot2018');

