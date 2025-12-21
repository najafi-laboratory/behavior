
clear vid
clear src
imaqreset

port = "COM4";      % <-- change to your COM port
baud = 115200;

s = serialport(port, baud);

% IMPORTANT: Arduino often resets when you open the port
pause(2.0);

vid = [];


% imaqhwinfo
% info = imaqhwinfo('pointgrey', 1);
% formats = info.SupportedFormats


vid = videoinput('pointgrey', 1, "F7_Raw8_640x512_Mode1");
src = getselectedsource(vid);

% propinfo(src, 'ExposureMode')
% src.ExposureMode = "Auto";
% src.ExposureMode = "Manual";
% src.ExposureMode = "Off";
% src.Exposure = 1;
% src.TriggerDelay = 1.13249 * 0.000001;
src.TriggerDelay = 0;
% propinfo(src, 'FrameRateMode')
src.FrameRateMode = "Manual";
% src.FrameRateMode = "Auto";
% src.FrameRateMode = "Off";
src.FrameRate = 250;
% src.Shutter = 0.01;
src.Gain = 10;
% propinfo(src, 'ShutterMode')
% src.ShutterMode = 'Manual';
% src.ShutterMode = 'Auto';
src.Shutter = 3;
% src.Shutter = 4;
% propinfo(src, 'Strobe3')
src.Strobe3 = 'On';
% src.Strobe3Delay = 0;
% src.Strobe3Duration = 0;
src.Strobe3Polarity = 'Low';

% triggerconfig(vid, "manual");
% vid.FramesPerTrigger = 1000;
% vid.TriggerRepeat = 6;
% vid.FramesPerTrigger = inf;
% vid.TriggerRepeat = 0;
vid.FramesPerTrigger = 1;
vid.TriggerRepeat = inf;

% triggerinfo(vid)
triggerCondition = "risingEdge";
% triggerSource = "externalTriggerMode0-Source2";
triggerSource = 'externalTriggerMode14-Source2';
% triggerSource = 'externalTriggerMode15-Source2';
% 
triggerconfig(vid, "hardware", triggerCondition, triggerSource);
% triggerconfig(vid, "manual", "none", "none");

% src.TriggerParameter = 1024;
% src.TriggerParameter = 400;
% src.TriggerParameter = 1;

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

% [data, time, metadata] = getdata(vid, vid.FramesAvailable);
% flushdata(vid)
for i = 1:3
    % trigger(vid);    
    % Send PWM ON command (0x01)
    write(s, uint8(1), "uint8");
    pause(1);  % let acquisition run
    write(s, uint8(2), "uint8");

    if i == 3
        stop(vid);
    end

    % Grab any frames acquired so far
    if vid.FramesAvailable > 0
        disp(['TrialNum: ' num2str(i)]);
        disp(['FramesAcquired so far: ' num2str(vid.FramesAcquired)])
        disp(['FramesAvailable: ' num2str(vid.FramesAvailable)]);
        % frames = getdata(vid, vid.FramesAvailable);
        [data, time, metadata] = getdata(vid, vid.FramesAvailable);
        
        % Optionally make sub-video
        subV = VideoWriter([folder + 'Trial' + num2str(i) + '.mp4'],'MPEG-4');
        open(subV);
        writeVideo(subV, data);
        close(subV);
    end

    ;
end
disp(['FramesAcquired so far: ' num2str(vid.FramesAcquired)]);

clear s

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





