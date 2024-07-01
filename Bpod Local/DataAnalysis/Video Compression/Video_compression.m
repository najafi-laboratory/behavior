% Load your original video
videoFile = 'E1VT_TrialVid_1_2024-07-01_145144.avi'; % Replace with your video file
video = VideoReader(videoFile);

% Get the original frame rate
originalFrameRate = video.FrameRate;

% Create a VideoWriter object with the desired format
outputFile = 'E1VT_TrialVid_1_2024-07-01_145144_compressed.avi'; % Replace with your output file name
writerObj = VideoWriter(outputFile, 'Motion JPEG AVI'); % Use 'Motion JPEG AVI' for MJPEG

% Set the frame rate to match the original video
writerObj.FrameRate = originalFrameRate;

% Set quality (for MPEG-4)
writerObj.Quality = 90; % Adjust quality as needed (0 to 100)

% Open the VideoWriter object
open(writerObj);

% Write each frame to the new file
while hasFrame(video)
    frame = readFrame(video);
    % Ensure frame dimensions are consistent with VideoWriter requirements
    % if size(frame, 1) <= 0 || size(frame, 2) <= 0
    %     error('Invalid frame dimensions.');
    % end    
    writeVideo(writerObj, frame);
end

% Close the VideoWriter object
close(writerObj);