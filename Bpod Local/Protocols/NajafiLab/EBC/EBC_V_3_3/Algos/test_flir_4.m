% Find available hardware
info = imaqhwinfo;

% Connect to a Video Input
% Camera (example with a 'gentl' device)
vid = videoinput('gentl', 1);

% Configure Video Source Properties
src = getselectedsource(vid);

% Find available info for connected Video Input
imaqhwinfo_vid = imaqhwinfo(vid);

% Set GPIO Line3 as Output
% src.LineSelector = "Line3";
% src.LineMode = "Output";

% Use callback function to generate logging file
% for sync
% sample gpio input lines when logging to get start sync and LED/puff sync.
% Log timestamp

% View the default logging mode.
currentLoggingMode = vid.LoggingMode;

% Configure the logging mode to disk.
vid.LoggingMode = 'disk';

% Verify the configuration.
currentLoggingMode = vid.LoggingMode;

% Create a VideoWriter object.
EBC_vid_log = VideoWriter('EBC_vid.mp4', 'MPEG-4');

% Configure the video input object to use the VideoWriter object.
vid.DiskLogger = EBC_vid_log;

% Start an acquisition
start(vid);

% Wait for the acquisition to finish.
wait(vid, 5)

% When logging large amounts of data to disk, disk writing occasionally lags behind the acquisition. To determine whether all frames are written to disk, you can optionally use the DiskLoggerFrameCount property.
while (vid.FramesAcquired ~= vid.DiskLoggerFrameCount)
    vid.FramesAcquired
    vid.DiskLoggerFrameCount
    pause(.1)
end

% Total number of frames that the object has acquired, regardless of how many frames have been extracted from the memory buffer, specified as a nonnegative integer.
vid.FramesAcquired

% Check available frames in video object
vid.FramesAvailable

% Capture the current frame
frame = getsnapshot(vid);

% Show frame
imshow(frame)


% Stop the acquisition
stop(vid);

% Clean up
delete(vid);
imaqreset;
clear;



vidTimer = [];
% Timer Routines to Incorporate
if isempty(vidTimer) || ~isvalid(vidTimer)
    % Create a timer object to grab frames
    vidTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.1, 'TimerFcn', @updateVideo);
    start(vidTimer);  % Start the timer
end


if ~isempty(vidTimer) && isvalid(vidTimer)
    stop(vidTimer);  % Stop the timer
    delete(vidTimer);  % Delete the timer object
    vidTimer = [];
end