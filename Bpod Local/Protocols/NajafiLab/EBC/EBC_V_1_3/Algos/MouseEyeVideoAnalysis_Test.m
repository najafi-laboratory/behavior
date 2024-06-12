MEV = MouseEyeVideoAnalysis_V_2;



% Function to record video during Trial
   % function recordVideoDuringTrial();
   %      startTime = ledOnset - 1; % Start recording 1s before LED onset
   %      stopTime = puffOffset + 2; % Stop recording 2s after puff offset
   %      trialVideoFrames = {};
   %      video.CurrentTime = startTime;
   %      currentTrialData = [];
   %      currentTrialTimes = [];
   %      while hasFrame(video) && video.CurrentTime <= stopTime
   %         frame = readFrame(video);
   %          trialVideoFrames{end+1} = frame;
   %          currentTime = video.CurrentTime;
   %          binFrame = imbinarize(rgb2gray(frame), graythresh(rgb2gray(frame)));
   %          imshow(frame, 'Parent', axOriginal); % Display original frame
   %          imshow(binFrame, 'Parent', axThresholded); % Display binary frame