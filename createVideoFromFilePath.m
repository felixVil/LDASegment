function createVideoFromFilePath(fileNameArray, videoFilePath)
numOfImages = numel(fileNameArray);
fileNameArray = sort(fileNameArray);
videoWriteObj = VideoWriter(videoFilePath);
open(videoWriteObj);
for k = 1:numOfImages
    filepath = fileNameArray{k};
    img = imread(filepath);
    text_str = sprintf('frame number = %04d', k);
    overlaidImg = insertText(img,[10 20],text_str,'FontSize',18,'BoxColor',...
    'green','BoxOpacity',0.4,'TextColor','white');
    writeVideo(videoWriteObj, overlaidImg);
end
close(videoWriteObj);
end