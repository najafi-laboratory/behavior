%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Define the video file path and initialize Bpod:

videoFile = 'C:\localscratch\behavior\behavior_setup\Bpod Local\Protocols\NajafiLab\EBC\EBC_V_1_0\videos\MyROI_short1.mp4'; 
% Initialize Bpod system
global BpodSystem
BpodSystem.ProtocolSettings = struct; % Load default settings
BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Load the ROI-framed video:
videoReader = VideoReader(videoFile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Create a VideoWriter object to save the processed frames (optional):
outputProcessedVideo = VideoWriter('C:\localscratch\behavior\behavior_setup\Bpod Local\Protocols\NajafiLab\EBC\EBC_V_1_0\videos\OutputTest3.mp4', 'MPEG-4');
open(outputProcessedVideo);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Predefine some variables:
totalFrames = floor(videoReader.Duration * videoReader.FrameRate);
fecValues = zeros(totalFrames, 1);
frameIndex = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Process each frame:

while hasFrame(videoReader)
    frame = readFrame(videoReader);
    
    % Convert the frame to grayscale
    grayFrame = rgb2gray(frame);
    
    % Apply median filtering to reduce noise
    filteredFrame = medfilt2(grayFrame);
    
    % Threshold the image to convert it into a binary image
    binaryFrame = imbinarize(filteredFrame);
    
    % Convert binary frame to uint8
    binaryFrame = uint8(binaryFrame) * 255;

    % Calculate the Fraction of Eyelid Closure (FEC)
    furArea = sum(binaryFrame(:));
    
    % Normalize the pixel counts to obtain the FEC
    fec = furArea / numel(binaryFrame);
    fecValues(frameIndex) = fec;
    
    % Real-time Display (Optional)
    % imshow(binaryFrame);
    % title(sprintf('Frame %d - FEC: %.2f', frameIndex, fec));
    % drawnow;
    
    % Save the processed frame to the output video (optional)
    writeVideo(outputProcessedVideo, binaryFrame);
    
    frameIndex = frameIndex + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Close the VideoWriter object:
close(outputProcessedVideo);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Save the FEC values to a file (optional):
save('fec_values.mat', 'fecValues');

%Display a message when done:
disp('Processing and FEC calculation completed.');