
vid = [];

if isempty(vid) || ~isvalid(vid)
    vid = videoinput('pointgrey', 1, "F7_Raw8_640x512_Mode1");
    src = getselectedsource(vid);
else
    stop(vid);
end

src.ExposureMode = "Auto";
src.TriggerDelay = 1.13249 * 0.000001;
src.FrameRateMode = "Manual";
src.FrameRate = 473.6355;
src.Shutter = 0.01;
src.Strobe3 = 'On';
triggerconfig(vid, "manual");
vid.FramesPerTrigger = 1000;
vid.TriggerRepeat = 0;

events = vid.EventLog;

vid.LoggingMode = 'disk&memory';
vid.DiskLogger = VideoWriter([".\Videos\TrialVideos\vid1" ],'MPEG-4');

start(vid);
trigger(vid);

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






