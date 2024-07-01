vid = [];


    % vid = videoinput('pointgrey', 1, "F7_Raw8_640x512_Mode1");
    vid = videoinput('gentl', 1, "Mono8");
    src = getselectedsource(vid);


% src.ExposureMode = "Auto";
src.TriggerDelay = 1.13249 * 0.000001;
% src.FrameRateMode = "Manual";
src.AcquisitionFrameRate = 4.747812500000000e+02;
% src.FrameRate = 473.6355;
% src.Shutter = 0.01;
% src.Strobe3 = 'On';
triggerconfig(vid, "manual");
vid.FramesPerTrigger = 500;
vid.TriggerRepeat = 0;

src.LineSelector = 'Line3';
src.LineMode = 'Output';
src.LineStatus =  'False';

camEvents = vid.EventLog;

vid.LoggingMode = "memory";

eventLog = vid.EventLog;

start(vid);
trigger(vid)

% while strcmp(src.LineStatus, 'False')
%     i = i + 1;
%     vid.FramesAvailable
%     pause(0.001);
% end

while vid.FramesAvailable < 500
    pause(0.001);
end

if vid.FramesAvailable > 0
    % disp([' FramesAvailable ' num2str(vid.FramesAvailable)])
    numFrames = vid.FramesAvailable;
    [data, time, metadata] = getdata(vid, numFrames);
    % frame = data(:,:,:,end);
    checkFps = 1;
    if checkFps == 1
        timeCheck = time - time(1);
        Ts = diff(time);
        avgFps_MeanBased = 1/mean(Ts)
        avgFps_NumFramesPerVidTime = length(time)/time(end)
    end
end

i = 1;
while strcmp(src.LineStatus, 'False')
    i = i + 1;
    pause(0.001);
end
i
disp('input detect')


clear vid
clear src
imaqreset
% pause(3);
% 
% [data, time, metadata] = getdata(vid, vid.FramesAvailable);
% 
% 
% 
% 
% stop(vid);
