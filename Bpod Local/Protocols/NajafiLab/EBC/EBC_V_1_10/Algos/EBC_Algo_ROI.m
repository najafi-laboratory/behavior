

 %1_Load the video file:
 videoFile = 'C:\localscratch\behavior\behavior_setup\Bpod Local\Protocols\NajafiLab\EBC\EBC_V_1_0\videos\OutputTest3.mp4';
 videoReader = VideoReader(videoFile);


 %2_Read the first frame and display it:
 frame = readFrame(videoReader);
 figure; imshow(frame);


 %3_Select the ROI interactively:**
 
 roi = drawellipse('Color','r');
  
 % Use INPUT to pause before starting frame reading loop
 input("Press ENTER to select ROI.");

 %4_Extract the coordinates of the ROI:
 % position = round(roi.Position); % [x, y, width, height]
 mask = uint8(createMask(roi));
 mask = repmat(mask, [1, 1, size(frame, 3)]);
 imshow(mask) % show mask
 
 %5_Create a VideoWriter object to save the new video:**
 outputVideo = VideoWriter('C:\localscratch\behavior\behavior_setup\Bpod Local\Protocols\NajafiLab\EBC\EBC_V_1_0\videos\ROIoutput.mp4', 'MPEG-4');
 open(outputVideo);

 %6_Loop through the video to extract and save the ROI from each frame:**
 while hasFrame(videoReader)

     frame = readFrame(videoReader);
     % Extract the ROI
  
     % roiFrame = imcrop(frame, position);
     roiFrame = frame.*mask;
     imshow(roiFrame) % show masked image

     % Write the ROI frame to the new video

     writeVideo(outputVideo, roiFrame);

 end

 %7_Close the VideoWriter object:**
 close(outputVideo);
   
 %8_Clean up and display completion message:**
 clear videoReader outputVideo frame roi roiFrame position;
 disp('ROI extraction and video creation complete.');