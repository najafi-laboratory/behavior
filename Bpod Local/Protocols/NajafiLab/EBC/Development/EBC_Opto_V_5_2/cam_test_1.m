
clear vid
clear src
imaqreset

vid = [];


% info = imaqhwinfo('pointgrey', 1);
% formats = info.SupportedFormats

vid = videoinput('pointgrey', 1, "F7_Raw8_640x512_Mode1");
src = getselectedsource(vid);


% src.ExposureMode = "Auto";
% src.TriggerDelay = 1.13249 * 0.000001;
src.TriggerDelay = 0;
src.FrameRateMode = "Manual";
src.FrameRate = 250;
% src.Shutter = 0.01;
src.Gain = 10;
src.Shutter = 3;
src.Strobe3 = 'On';
% triggerconfig(vid, "manual");
vid.FramesPerTrigger = 1000;
vid.TriggerRepeat = 6;

% triggerinfo(vid)
triggerCondition = "risingEdge";
% triggerSource = "externalTriggerMode0-Source2";
triggerSource = 'externalTriggerMode14-Source2';
% triggerSource = 'externalTriggerMode15-Source2';

triggerconfig(vid, "hardware", triggerCondition, triggerSource);
triggerconfig(vid, "manual", "none", "none");

events = vid.EventLog;

% vid.LoggingMode = 'memory';
% vid.LoggingMode = 'disk';
vid.LoggingMode = 'disk&memory';
folder = ".\Videos\TrialVideos\";
if ~exist(folder,'dir')
    mkdir(folder);
end
vid.DiskLogger = VideoWriter([folder + "vid1"],'MPEG-4');

start(vid)

for i = 1:3
    trigger(vid);    
    pause(7);  % let acquisition run

    % Grab any frames acquired so far
    if vid.FramesAvailable > 0
        disp(['TrialNum: ' num2str(i)]);
        disp(['FramesAvailable: ' num2str(vid.FramesAvailable)]);
        frames = getdata(vid, vid.FramesAvailable);
        
        % Optionally make sub-video
        subV = VideoWriter([folder + 'Trial' + num2str(i) + '.mp4'],'MPEG-4');
        open(subV);
        writeVideo(subV, frames);
        close(subV);
    end

    disp(['FramesAcquired so far: ' num2str(vid.FramesAcquired)]);
end



stop(vid)

disp([' is running ' num2str(isrunning(vid))])

while (vid.FramesAvailable > 0)
    disp([' is running ' num2str(isrunning(vid))])
end

pause(1)

stop(vid)

while (vid.FramesAcquired ~= vid.DiskLoggerFrameCount)   
    disp(['FramesAcquired ' num2str(vid.FramesAcquired) '  DiskLoggerFrameCount ' num2str(vid.DiskLoggerFrameCount)])
    pause(0.1)
end

[data, time, metadata] = getdata(vid, vid.FramesAvailable);
    
timeCheck = time - time(1);
Ts = diff(time);
avgFps_MeanBased = 1/mean(Ts)
avgFps_NumFramesPerVidTime = length(time)/time(end)
    

disp([' is running ' num2str(isrunning(vid))])

vid.DiskLogger = VideoWriter([".\Videos\TrialVideos\vid2" ],"Grayscale AVI");

disp('---------------------')
disp('VID 2')
disp('---------------------')

start(vid);
trigger(vid)

pause(1)

stop(vid)


while (vid.FramesAcquired ~= vid.DiskLoggerFrameCount) 
    disp(['FramesAcquired ' num2str(vid.FramesAcquired) '  DiskLoggerFrameCount ' num2str(vid.DiskLoggerFrameCount)])
    pause(0.1)
end

clear vid
clear src
imaqreset





