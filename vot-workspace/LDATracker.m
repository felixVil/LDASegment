function LDATracker
global kerasTrackerPath;
global pythonInterpreterPath;

pythonInterpreterPath = 'D:\\Felix\\WPy64-3680\\scripts\\python.bat';
kerasTrackerPath = 'D:\Felix\LDASegment\KerasTracker';

regionJsonFileInit = fullfile(kerasTrackerPath, 'regionData.json');
regionJsonFileUpdate = fullfile(kerasTrackerPath, 'regionDataUpdate.json'); 
while (exist(regionJsonFileUpdate, 'file') == 2);end %Wait for previous run to finish.


signalFilePath = fullfile(kerasTrackerPath, 'info.txt');
if (exist(signalFilePath, 'file') == 2)
    delete(signalFilePath);
end

[handle, image, region] = vot('polygon');
% 
%converting region to region json
%regionStruct = struct('x', region(1),'y', region(2), 'width', region(3), 'height', region(4));
text = jsonencode(region);

fileID = fopen(regionJsonFileInit,'w');
fwrite(fileID, text);
fclose(fileID);

% Initialize the tracker
ldaTrack_initialize(image, regionJsonFileInit);

while true
    
    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);
    
    if isempty(image)
        ldaTrack_update('', true);
        break;
    end
    
    % Perform a tracking step, obtain new region
    [region, confidence] = ldaTrack_update(image, false);
    
    % **********************************
    % VOT: Report position for frame
    % **********************************
    handle = handle.report(handle, region, confidence);
    
end

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle);

end

function ldaTrack_initialize(imageFilePath, regionJsonFile)
global kerasTrackerPath;
global pythonInterpreterPath;
commandStr = sprintf('%s "%s" "%s" "%s" &', pythonInterpreterPath, fullfile(kerasTrackerPath, 'lda_tracker_init.py'), imageFilePath, regionJsonFile);
system(commandStr);
end

function [region, confidence] = ldaTrack_update(imageFilePath, isTerminate)
text = '';
global kerasTrackerPath;
regionJsonFileUpdate = fullfile(kerasTrackerPath, 'regionDataUpdate.json'); 
signalFilePath = fullfile(kerasTrackerPath, 'info.txt');
fileID = fopen(signalFilePath,'w');
if isTerminate
    fprintf(fileID, 'EndOfSequence\n');
else
    fprintf(fileID, '%s', imageFilePath);
end
fclose(fileID);
while ~(exist(regionJsonFileUpdate, 'file') == 2);end %Wait for tracker to react.
while ~contains(text, ':')
    fileID = fopen(regionJsonFileUpdate, 'r');
    text = fread(fileID, '*char')';
    fclose(fileID);
end
fprintf('text in file : %s is: %s\n', imageFilePath, text);
region = jsondecode(text);
region = region.region;
region = cellfun(@str2double, region);
region = region';
confidence = 1;
fprintf('rect for image : %s is:\n', imageFilePath);
disp(region);
while (exist(signalFilePath, 'file') == 2)
    pause(1);
    delete(signalFilePath);
end
while (exist(regionJsonFileUpdate, 'file') == 2)
    pause(1);
    delete(regionJsonFileUpdate);
end
end