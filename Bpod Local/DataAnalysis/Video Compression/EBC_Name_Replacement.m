% Define the directory containing the videos
inputDir = 'C:\behavior\video_data\E5LG\250123'; % Replace with your directory
outputDir = 'C:\behavior\video_data\E4L7\250123'; % Replace with your output directory
% Get a list of all video files in the directory
videoFiles = dir(fullfile(inputDir, '*.avi')); % Adjust the file extension if needed
% Loop over each video file
for i = 1:length(videoFiles)
    % Get the original video file path
    originalFile = fullfile(inputDir, videoFiles(i).name);
    % Create a new name for the output file
    originalName = videoFiles(i).name;
    newName = strrep(originalName, 'E5LG', 'E4L7');
    outputFile = fullfile(outputDir, newName);
    % Rename the file by copying it to the new location with the new name
    copyfile(originalFile, outputFile);
    % Display progress
    fprintf('Renamed video %d of %d: %s to %s\n', i, length(videoFiles), originalName, newName);
end
fprintf('All videos renamed and saved in %s\n', outputDir);