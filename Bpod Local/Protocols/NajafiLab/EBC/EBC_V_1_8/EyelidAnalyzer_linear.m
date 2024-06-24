% axOriginal
% axThresholded
% axFEC
% 
% binFrame
% binaryVideos
% bloadVideoAndDrawROI
% bselectROI
% bstartVideo
% bstopVideo
% 
% currentFrame
% 
% DataQueue
% 
% EBC_vid_log_trial
% eyeOpenMax
% 
% fec
% fecData        
% fecPlot
% fecTimes
% fpsCheck = 0;
% fps
% fpara
% frame
% 
% guiSettings
% 
% hFig
% 
% isPaused = true;
% isPlaying = false; 
% isZoom = false;
% 
% maxOpen = 0;
% mask
% 
% roiHandle
% roiPosition
% 
% sliderThreshold
% sliderEyeOpen
% src
% 
% tAdjustThresh
% threshold = 100;
% thresholdEyeOpen
% tMinThresh
% trialVideoDir = '.\Videos\TrialVideos\';
% 
% vid        
% video
% % vidWriter
% vidTimer
% 
% startTime  % To store the start time of the video
% stopTime

vidTimer = [];

createGUI

startGUIVideo

% https://www.mathworks.com/help/parallel-computing/perform-data-acquisition-in-parallel-with-postprocessing.html
% parallel pool with 1 worker
parpool('Processes',1);
% To send information back from the worker to the MATLAB client, create a DataQueue object.
DataQueue = parallel.pool.DataQueue;
%  To display images every time they arrive from the DataQueue object, use afterEach.
afterEach(DataQueue, @updateVideoPara);
% freq = 5;
freq = 5;
fpara = parfeval(@getFrameFromCamera,0,DataQueue,freq);

function createGUI()
    % console warning for java.awt.Component subclass method call automatically
    % delegated to the Event Dispatch Thread (EDT)
    % Create the main figure
    hFig = figure('Name', 'Mouse Eye Video Analysis', 'Position', [12 324 961 666], 'NumberTitle', 'off', 'Resize', 'on', 'CloseRequestFcn', @onGUIClose);
    
    % Create axes for displaying the original and thresholded videos
    axOriginal = subplot(2, 2, 1, 'Parent', hFig);
    title(axOriginal, 'Original Video');
    axThresholded = subplot(2, 2, 2, 'Parent', hFig);
    title(axThresholded, 'Thresholded Video');
    
    % Create a plot for Fraction of Eyelid Closure (FEC)
    axFEC = subplot(2, 1, 2, 'Parent', hFig);
    title(axFEC, 'Fraction of Eyelid Closure Over Time');
    xlabel(axFEC, 'Time (s)');
    ylabel(axFEC, 'FEC (%)');
    hold(axFEC, 'on');
    fecPlot = plot(axFEC, NaN, NaN, 'b-'); % Initialize FEC plot
    
    % % Add controls and buttons
    sliderThreshold = uicontrol('Style', 'slider', 'Min', 0, 'Max', 255, 'Value', 100, ...
        'Position', [150 50 120 20], 'Callback', @adjustThreshold);
    sliderEyeOpen = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', 0.7, ...
        'Position', [150 30 120 20], 'Callback', @adjustEyeOpenThreshold);
    tAdjustThresh = uicontrol('Style', 'text', 'Position', [20 50 120 20], 'String', 'Adjust Threshold:');
    tMinThresh = uicontrol('Style', 'text', 'Position', [20 30 200 20], 'String', 'Minimum Eye Openness Threshold:');
    bstartVideo = uicontrol('Style', 'pushbutton', 'String', 'Start', ...
        'Position', [300 20 50 20], 'Callback', @startGUIVideo);
    bstopVideo = uicontrol('Style', 'pushbutton', 'String', 'Stop', ...
        'Position', [360 20 50 20], 'Callback', @stopGUIVideo);
    % bloadVideoAndDrawROI = uicontrol('Style', 'pushbutton', 'String', 'Load Video & Draw ROI', ...
    %     'Position', [420 20 150 20], 'Callback', @loadVideoAndDrawROI);

    bselectROI = uicontrol('Style', 'pushbutton', 'String', 'Select ROI', 'Units', 'normalized',...
        'Position', [0.25, 0.25, 0.15, 0.05], 'Callback', @selectROI);
    uicontrol('Parent', hFig, 'Style', 'pushbutton', 'String', 'Pause', ... 
        'Units', 'normalized', 'Position', [0.25, 0.15, 0.15, 0.05], 'Callback', @pauseVideo);
                        
    % Load GUI settings if exist
    if exist('guiSettings.mat', 'file')
        load('guiSettings.mat', 'guiSettings');
        guiSettings = guiSettings;
        % Set the GUI components to last saved states
        set(sliderThreshold, 'Value', guiSettings.thresholdValue);
        set(sliderEyeOpen, 'Value', guiSettings.eyeOpenThreshold);
    end

    %warning('on','verbose')
    warning('off','imaq:getdata:infFramesPerTrigger');
    warning('off','imaq:gige:adaptorErrorPropSet');
    warning('off','MATLAB:JavaEDTAutoDelegation');
    
end

function onGUIClose(~, ~)

    close(EBC_vid_log_trial);

    poolobj = gcp('nocreate');
    delete(poolobj);

    stopGUIVideo(); % Ensure the live video is stopped
    saveSettings(); % Save settings before closing
    delete(hFig); % Close the figure
    % delete(timerfindall)
end

function startGUIVideo()          
    % Find available hardware
    % info = imaqhwinfo;
    vid = videoinput('pointgrey', 1);
    src = getselectedsource(vid);
    % src.FrameRate = 30;
    
    % Connect to a Video Input Camera
    % vid = videoinput('gentl', 1);
    
    % Configure Video Source Properties
    % src = getselectedsource(vid);

    % check fps of camera
    fps = src.FrameRate;
    % if fpsCheck
    %     if((fps > 30) && (fps < 31))
    %         %disp(['Set camera FPS = 30']);
    %         onGUIClose;
    %         MatExc = MException('MyComponent:noSuchVariable','Set camera FPS to 30 fps. fps = %s', num2str(fps));
    %         throw(MatExc);               
    %     end
    % end           
    % Set trigger config
    vid.FramesPerTrigger = inf;

    % Start the acquisition
    start(vid);

    while ~isrunning(vid)
        disp(['vid not started']);
    end
   
    pause(0.1);

    disp(['Video Acquisition Started']);

    isPaused = false;        

    % % Record the start time
    startTime = datetime('now');
    
    if isempty(vidTimer) || isvalid(vidTimer)
        vidTimer = timer('TimerFcn','', 'ExecutionMode', 'fixedRate', 'Period', 0.065);
        set(vidTimer, 'TimerFcn', @(x,y)getFrameFromCamera());
        start(vidTimer);
        disp(['Video Update Timer Started']);
    end  

end

function stopGUIVideo()
    if isempty(vidTimer) && isvalid(vidTimer)
        stop(vidTimer);
        delete(vidTimer);
        vidTimer = [];
    end
    
    if isempty(vid) && isvalid(vid)
        stop(vid);
        delete(vid);
    end

    % cancel(fpara);
    poolobj = gcp('nocreate');
    delete(poolobj);

    warning('off','MATLAB:JavaEDTAutoDelegation');
    imaqreset; % Reset Image Acquisition Toolbox
    clear vid; % Clear the video object

    disp(['Video Acquisition Stopped']);
end     

% pauseVideo
function pauseVideo()             
    if isPaused
        start(vidTimer)
    else                
        stop(vidTimer);
    end
    isPaused = isPaused; % Toggle pause state
end   

% Function to select ROI    
function selectROI()
    pauseVideo;
    if isempty(vid)
        msgbox('Start video first!');
        return;
    end

    roiHandle = imellipse(axOriginal); % Let user draw an ellipse
    roiPosition = getPosition(roiHandle);  % Get ROI position before playing  
    disp(['ROI Selected']);
    pauseVideo;
end

% update gui with roi-binarized image
function updateBinaryVideo()
    if isobject(roiHandle) && isvalid(roiHandle)
        mask = createMask(roiHandle); % Create a mask from the ROI
        grayFrame = im2gray(frame);
        grayFrame(~mask) = 0; % Apply mask to the frame
        binFrame = imbinarize(grayFrame, threshold / 255);
        imshow(binFrame, 'Parent', axThresholded); % Display the frame in the original video axes                     
        calculateFEC;
    else
        % disp('ROI object is invalid or deleted.');
    end
end       

% calculate fraction eye closure
function calculateFEC()
    totalEllipsePixels = numel(find(mask == 1));            
    openPixels =  sum(binFrame(:) == 1); % Assuming open eye corresponds to 1, but it's 0
    fec = (openPixels / totalEllipsePixels) * 100; % FEC percentage
    currentTime = seconds(datetime('now') - startTime);
    fecData = [fecData, fec];
    fecTimes = [fecTimes, currentTime];
    set(fecPlot, 'XData', fecTimes, 'YData', fecData);
end

function updateVideoPara(img)     
    if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
        obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before playing  
    end

    imshow(obj.frame, 'Parent', obj.axOriginal);

    if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
        setPosition(obj.roiHandle, obj.roiPosition);  % Reset the position for consistency
    elseif ~isempty(obj.roiPosition)
        obj.roiHandle = imellipse(obj.axOriginal, obj.roiPosition);  % Recreate ROI if not valid
    end

    obj.updateBinaryVideo;
end

function getFrameFromCamera(DataQueue, freq)
    if isrunning(vid)
        if vid.FramesAvailable > 0
            vid.FramesAvailable
            [data, time, metadata] = getdata(vid);
            frame = data(:,:,:,end);
            % frame = frame;
            if isZoom
                if isempty(frame)
                    img_crop = frame(409:550, 575:730);
                    frame = img_crop;
                    % imshow(frame, 'Parent', axOriginal);
                end
            else
                % imshow(frame, 'Parent', axOriginal);
            end
            send(DataQueue, obj);
            flushdata(vid);
        if isobject(roiHandle) && isvalid(roiHandle)
            setPosition(roiHandle, roiPosition);  % Reset the position for consistency
        elseif isempty(roiPosition)
            roiHandle = imellipse(axOriginal, roiPosition);  % Recreate ROI if not valid
        end

        % updateBinaryVideo;
        % stop(vidTimer);
        % start(vidTimer);                    
        end
    else
        % disp(['Video Input Buffer Empty']);
    end
end