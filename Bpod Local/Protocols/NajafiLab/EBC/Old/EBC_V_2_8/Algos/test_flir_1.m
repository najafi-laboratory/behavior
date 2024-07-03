





camera = videoinput("gentl", 1, "Mono8");


% Find available hardware
info = imaqhwinfo;
% Connect to a camera (example with a 'winvideo' device)
vid = videoinput('gentl', 1);
% Configure properties
src = getselectedsource(vid);



src.Exposure = -6;
% Preview the video
preview(vid);
% Start an acquisition
start(vid);







% Stop the acquisition
stop(vid);
% Clean up
delete(vid);
clear vid;


% img = snapshot(camera);  % Take a snapshot
% imshow(img, 'Parent', ax);  % Display the image

% Function to capture and display an image
function captureImage(~, ~)
    img = snapshot(cam);  % Take a snapshot
    imshow(img, 'Parent', ax);  % Display the image
end


% video = VideoReader('sampleVideo.mp4'); % Assuming video is predefined