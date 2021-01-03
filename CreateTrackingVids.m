function CreateTrackingVids(imageFolderPath)
resultImageFileInfo = dir(fullfile(imageFolderPath, 'res*'));
resultMaskFileInfo = dir(fullfile(imageFolderPath, 'mask*'));

imageFilesNames = {resultImageFileInfo.name}';
maskFilesNames = {resultMaskFileInfo.name}';

imageFilePaths = fullfile(imageFolderPath, imageFilesNames);
maskFilePaths = fullfile(imageFolderPath, maskFilesNames);

videoFolderPath = fullfile(imageFolderPath, 'videos');
if exist(videoFolderPath, 'dir') ~= 7
    mkdir(videoFolderPath);
end

trackVidPath = fullfile(videoFolderPath, 'trackVid.avi');
maskVidPath = fullfile(videoFolderPath, 'maskVid.avi');
createVideoFromFilePath(imageFilePaths, trackVidPath);
createVideoFromFilePath(maskFilePaths, maskVidPath);
end

