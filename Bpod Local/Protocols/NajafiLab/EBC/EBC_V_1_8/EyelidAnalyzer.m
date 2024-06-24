classdef EyelidAnalyzer < handle
    properties        
        axOriginal
        axThresholded
        axFEC

        binFrame
        binaryVideos
        bloadVideoAndDrawROI
        bselectROI
        bstartVideo
        bstopVideo

        currentFrame

        DataQueue

        EBC_vid_log_trial
        eyeOpenMax
        
        fec
        fecData        
        fecPlot
        fecTimes
        fpsCheck = 0;
        fps
        fpara
        frame

        guiSettings
        
        hFig

        isPaused = true;
        isPlaying = false; 
        isZoom = false;

        maxOpen = 0;
        mask
        
        roiHandle
        roiPosition

        sliderThreshold
        sliderEyeOpen
        src

        tAdjustThresh
        threshold = 100;
        thresholdEyeOpen
        tMinThresh
        trialVideoDir = '.\Videos\TrialVideos\';

        vid        
        video
        % vidWriter
        vidTimer

        startTime  % To store the start time of the video
        stopTime
    end
    
    methods
        function obj = EyelidAnalyzer()
            % Init Properties

            % Initialize data storage
            obj.fecData = [];
            obj.fecTimes = [];
            obj.binaryVideos = {};
            
            % Create GUI
            obj.createGUI();    
        end
        
        function createGUI(obj)
            % console warning for java.awt.Component subclass method call automatically
            % delegated to the Event Dispatch Thread (EDT)
            % Create the main figure
            obj.hFig = figure('Name', 'Mouse Eye Video Analysis', 'Position', [12 324 961 666], 'NumberTitle', 'off', 'Resize', 'on', 'CloseRequestFcn', @obj.onGUIClose);
            
            % Create axes for displaying the original and thresholded videos
            obj.axOriginal = subplot(2, 2, 1, 'Parent', obj.hFig);
            title(obj.axOriginal, 'Original Video');
            obj.axThresholded = subplot(2, 2, 2, 'Parent', obj.hFig);
            title(obj.axThresholded, 'Thresholded Video');
            
            % Create a plot for Fraction of Eyelid Closure (FEC)
            obj.axFEC = subplot(2, 1, 2, 'Parent', obj.hFig);
            title(obj.axFEC, 'Fraction of Eyelid Closure Over Time');
            xlabel(obj.axFEC, 'Time (s)');
            ylabel(obj.axFEC, 'FEC (%)');
            hold(obj.axFEC, 'on');
            obj.fecPlot = plot(obj.axFEC, NaN, NaN, 'b-'); % Initialize FEC plot
            
            % % Add controls and buttons
            obj.sliderThreshold = uicontrol('Style', 'slider', 'Min', 0, 'Max', 255, 'Value', 100, ...
                'Position', [150 50 120 20], 'Callback', @obj.adjustThreshold);
            obj.sliderEyeOpen = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', 0.7, ...
                'Position', [150 30 120 20], 'Callback', @obj.adjustEyeOpenThreshold);
            obj.tAdjustThresh = uicontrol('Style', 'text', 'Position', [20 50 120 20], 'String', 'Adjust Threshold:');
            obj.tMinThresh = uicontrol('Style', 'text', 'Position', [20 30 200 20], 'String', 'Minimum Eye Openness Threshold:');
            obj.bstartVideo = uicontrol('Style', 'pushbutton', 'String', 'Start', ...
                'Position', [300 20 50 20], 'Callback', @obj.startGUIVideo);
            obj.bstopVideo = uicontrol('Style', 'pushbutton', 'String', 'Stop', ...
                'Position', [360 20 50 20], 'Callback', @obj.stopGUIVideo);
            % obj.bloadVideoAndDrawROI = uicontrol('Style', 'pushbutton', 'String', 'Load Video & Draw ROI', ...
            %     'Position', [420 20 150 20], 'Callback', @obj.loadVideoAndDrawROI);

            obj.bselectROI = uicontrol('Style', 'pushbutton', 'String', 'Select ROI', 'Units', 'normalized',...
                'Position', [0.25, 0.25, 0.15, 0.05], 'Callback', @obj.selectROI);
            uicontrol('Parent', obj.hFig, 'Style', 'pushbutton', 'String', 'Pause', ... 
                'Units', 'normalized', 'Position', [0.25, 0.15, 0.15, 0.05], 'Callback', @obj.pauseVideo);
                                
            % Load GUI settings if exist
            if exist('guiSettings.mat', 'file')
                load('guiSettings.mat', 'guiSettings');
                obj.guiSettings = guiSettings;
                % Set the GUI components to last saved states
                set(obj.sliderThreshold, 'Value', obj.guiSettings.thresholdValue);
                set(obj.sliderEyeOpen, 'Value', obj.guiSettings.eyeOpenThreshold);
            end

            %warning('on','verbose')
            warning('off','imaq:getdata:infFramesPerTrigger');
            warning('off','imaq:gige:adaptorErrorPropSet');
            warning('off','MATLAB:JavaEDTAutoDelegation');
            
            % https://www.mathworks.com/help/parallel-computing/perform-data-acquisition-in-parallel-with-postprocessing.html
            % parallel pool with 1 worker
            parpool('Processes',1);
            % To send information back from the worker to the MATLAB client, create a DataQueue object.
            obj.DataQueue = parallel.pool.DataQueue;
            %  To display images every time they arrive from the DataQueue object, use afterEach.
            afterEach(obj.DataQueue, @updateVideoPara);
            % freq = 5;

        end

        % connect video input
        function connectVideoTrial(obj, ~, ~)            
            % set strobe using pointgrey
            % v = videoinput("pointgrey", 1, "F7_Mono8_1280x1024_Mode0");
            % v.ROIPosition = [0 0 1280 1024];
            obj.vid = videoinput('pointgrey', 1);
            obj.src = getselectedsource(obj.vid);
            % obj.src.Exposure = 1.624512e+00;
            obj.src.ExposureMode = "Auto";
            % obj.src.FrameRate = 3.019109e+01;
            % obj.src.Sharpness = 512;
            %obj.src.SharpnessMode = "Auto";
            % obj.src.FrameRate = 30;
            obj.src.Shutter = 2;
            obj.src.Strobe3 = 'On';
            % delete(obj.vid);
            obj.src.FrameRateMode = "Off";

            % obj.vid = videoinput('gentl', 1);     
            disp(['Video Input Connected']);
            % Configure Video Source Properties
            obj.src = getselectedsource(obj.vid);

            % check fps of camera
            % obj.fps = obj.src.AcquisitionFrameRate;
            obj.fps = obj.src.FrameRate;

            % obj.fpsCheck = 0;
            % if obj.fpsCheck
            %     if ~((obj.fps > 30) && (obj.fps < 31))
            %         %disp(['Set camera FPS = 30']);
            %         obj.onGUIClose;
            %         MatExc = MException('MyComponent:noSuchVariable','Set camera FPS to 30 fps. fps = %s', num2str(obj.fps));
            %         throw(MatExc);               
            %     end
            % end
            framesPerTrigger = 1;
            % numTriggers = 200;
            numTriggers = Inf;
            triggerCondition = "risingEdge";
            triggerSource = "externalTriggerMode0-Source2";
            
            triggerconfig(obj.vid, "hardware", triggerCondition, triggerSource);
            obj.vid.FramesPerTrigger = framesPerTrigger;
            obj.vid.TriggerRepeat = numTriggers - 1;

            %https://github.com/cortex-lab/mmmGUI

            % propinfo(obj.vid,'TriggerType')

            % GPIO Input Line 2 settings
            % obj.vid.TriggerType = 'hardware';
            if 0
                obj.src.LineSelector = 'Line2';
                obj.src.LineMode = 'Input';
    
                % Configure the trigger settings.
                %triggerinfo(obj.vid)
                % triggerconfig(obj.vid, 'hardware','risingEdge','TTL');
                triggerconfig(obj.vid, 'hardware');
                
                obj.src.TriggerMode = 'On';
                obj.src.TriggerActivation = 'RisingEdge';
                
                obj.src.TriggerDelayEnabled = 'False';
                obj.src.TriggerSelector = 'FrameStart';
                obj.vid.FramesPerTrigger = inf;
            

                % GPIO Output Line 3 settings
                obj.src.LineSelector = 'Line3';
                obj.src.LineMode = 'Output';
            end

            obj.trialVideoDir = '.\Videos\TrialVideos\';
            [status, msg, msgID] = mkdir(obj.trialVideoDir);
        end

        % disconnect video input
        function disconnectVideoTrial(obj, ~, ~)    
            if ~isempty(obj.vid) && isvalid(obj.vid)
                delete(obj.vid);
            end
            
            imaqreset; % Reset Image Acquisition Toolbox
            clear obj.vid; % Clear the video object
            disp(['Video Input Disconnected']);
        end  

        function startGUIVideo(obj, ~, ~)          
            % Find available hardware
            % info = imaqhwinfo;
            obj.vid = videoinput('pointgrey', 1);
            obj.src = getselectedsource(obj.vid);
            % obj.src.FrameRate = 30;
            
            % Connect to a Video Input Camera
            % obj.vid = videoinput('gentl', 1);
            
            % Configure Video Source Properties
            % obj.src = getselectedsource(obj.vid);

            % check fps of camera
            obj.fps = obj.src.FrameRate;
            if obj.fpsCheck
                if ~((obj.fps > 30) && (obj.fps < 31))
                    %disp(['Set camera FPS = 30']);
                    obj.onGUIClose;
                    MatExc = MException('MyComponent:noSuchVariable','Set camera FPS to 30 fps. fps = %s', num2str(obj.fps));
                    throw(MatExc);               
                end
            end           
            % Set trigger config
            obj.vid.FramesPerTrigger = inf;

            % Start the acquisition
            start(obj.vid);

            while ~isrunning(obj.vid)
                disp(['vid not started']);
            end

            pause(.1);

            disp(['Video Acquisition Started']);

            obj.isPaused = false;        

            % % Record the start time
            obj.startTime = datetime('now');
            
            % Create and start a timer to update the video display
            if 0
                if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
                    obj.vidTimer = timer('TimerFcn','', 'ExecutionMode', 'fixedRate', 'Period', 0.065);
                    set(obj.vidTimer, 'TimerFcn', @(x,y)obj.updateVideo());
                    start(obj.vidTimer);
                    disp(['Video Update Timer Started']);
                end            
            end
            freq = 5;
            obj.fpara = parfeval(@getFrameFromCamera,0,obj.DataQueue,freq);

            if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
                obj.vidTimer = timer('TimerFcn','', 'ExecutionMode', 'fixedRate', 'Period', 0.065);
                set(obj.vidTimer, 'TimerFcn', @(x,y)obj.getFrameFromCamera());
                start(obj.vidTimer);
                disp(['Video Update Timer Started']);
            end  

        end

        function updateVideoPara(obj, ~, ~)     
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

        function getFrameFromCamera(obj, DataQueue, freq)


            if isrunning(obj.vid)
                if obj.vid.FramesAvailable > 0
                    obj.vid.FramesAvailable
                    [data, time, metadata] = getdata(obj.vid);
                    obj.frame = data(:,:,:,end);
                    frame = obj.frame;
                    if obj.isZoom
                        if ~isempty(obj.frame)
                            img_crop = obj.frame(409:550, 575:730);
                            obj.frame = img_crop;
                            % imshow(obj.frame, 'Parent', obj.axOriginal);
                        end
                    else
                        % imshow(obj.frame, 'Parent', obj.axOriginal);
                    end
                    send(obj.DataQueue, obj);
                    flushdata(obj.vid);
                if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
                    setPosition(obj.roiHandle, obj.roiPosition);  % Reset the position for consistency
                elseif ~isempty(obj.roiPosition)
                    obj.roiHandle = imellipse(obj.axOriginal, obj.roiPosition);  % Recreate ROI if not valid
                end
    
                % obj.updateBinaryVideo;
                stop(obj.vidTimer);
                start(obj.vidTimer);                    
                end
            else
                % disp(['Video Input Buffer Empty']);
            end



        end

        function stopGUIVideo(obj, ~, ~)
            if ~isempty(obj.vidTimer) && isvalid(obj.vidTimer)
                stop(obj.vidTimer);
                delete(obj.vidTimer);
                obj.vidTimer = [];
            end
            
            if ~isempty(obj.vid) && isvalid(obj.vid)
                stop(obj.vid);
                delete(obj.vid);
            end

            % cancel(obj.fpara);
            poolobj = gcp('nocreate');
            delete(poolobj);

            warning('off','MATLAB:JavaEDTAutoDelegation');
            imaqreset; % Reset Image Acquisition Toolbox
            clear obj.vid; % Clear the video object

            disp(['Video Acquisition Stopped']);
        end     

        % pauseVideo
        function pauseVideo(obj, ~, ~)             
            if obj.isPaused
                start(obj.vidTimer)
            else                
                stop(obj.vidTimer);
            end
            obj.isPaused = ~obj.isPaused; % Toggle pause state
        end    

        % Function to select ROI    
        function selectROI(obj, ~, ~)
            obj.pauseVideo;
            if isempty(obj.vid)
                msgbox('Start video first!');
                return;
            end

            obj.roiHandle = imellipse(obj.axOriginal); % Let user draw an ellipse
            obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before playing  
            disp(['ROI Selected']);
            obj.pauseVideo;
        end

        function startVideoTrial(obj, ~, ~) 
            obj.isPaused = false;
            %disp(['Video Trial Started']);
            start(obj.vid);        
        end       

        function stopVideoTrial(obj, ~, ~) 
            stop(obj.vid);
            obj.isPaused = true;
        end

        function processVideoTrial(obj, currentTrial, subjectName)
            % open(obj.EBC_vid_log_trial);  
            %disp(['Process Video']);
            while obj.vid.FramesAvailable > 0
                disp(['FramesAvailable', num2str(obj.vid.FramesAvailable)]);
                [data, time, metadata] = getdata(obj.vid, obj.vid.FramesAvailable);

                timeCheck = 0;
                if timeCheck
                    % check timing
                    timeCheck = time - time(1);
                    figure();
                    montage(data);
                end

                obj.frame = data(:,:,:,end);
                currentTrial = sprintf( '%04d', currentTrial);
                videoFilename = [obj.trialVideoDir, subjectName, '_TrialVid_', currentTrial, '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.mp4'];
                obj.EBC_vid_log_trial = VideoWriter(videoFilename, 'MPEG-4');                     
                open(obj.EBC_vid_log_trial);
                % update with binary vid?
                writeVideo(obj.EBC_vid_log_trial, data);
                % crop video to zoomed portion
                if obj.isZoom
                    if ~isempty(obj.frame)
                        img_crop = obj.frame(409:550, 575:730);
                        obj.frame = img_crop;
                        imshow(obj.frame, 'Parent', obj.axOriginal);
                    end
                else
                    imshow(obj.frame, 'Parent', obj.axOriginal);
                end
                close(obj.EBC_vid_log_trial);
                obj.calculateFEC;
            end
             
            % flushdata(obj.vid);                   
        end    
     
        % update gui with most recent frame from image buffer
        function updateVideo(obj, ~, ~)     
            
            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before playing  
            end

            if isrunning(obj.vid)
                if obj.vid.FramesAvailable > 0
                    obj.vid.FramesAvailable
                    [data, time, metadata] = getdata(obj.vid);
                    obj.frame = data(:,:,:,end);
                    if obj.isZoom
                        if ~isempty(obj.frame)
                            img_crop = obj.frame(409:550, 575:730);
                            obj.frame = img_crop;
                            imshow(obj.frame, 'Parent', obj.axOriginal);
                        end
                    else
                        imshow(obj.frame, 'Parent', obj.axOriginal);
                    end
                    flushdata(obj.vid);
                if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
                    setPosition(obj.roiHandle, obj.roiPosition);  % Reset the position for consistency
                elseif ~isempty(obj.roiPosition)
                    obj.roiHandle = imellipse(obj.axOriginal, obj.roiPosition);  % Recreate ROI if not valid
                end
    
                obj.updateBinaryVideo;
                stop(obj.vidTimer);
                start(obj.vidTimer);                    
                end
            else
                % disp(['Video Input Buffer Empty']);
            end


        end

        % update gui with roi-binarized image
        function updateBinaryVideo(obj)
            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
                obj.mask = createMask(obj.roiHandle); % Create a mask from the ROI
                grayFrame = im2gray(obj.frame);
                grayFrame(~obj.mask) = 0; % Apply mask to the frame
                obj.binFrame = imbinarize(grayFrame, obj.threshold / 255);
                imshow(obj.binFrame, 'Parent', obj.axThresholded); % Display the frame in the original video axes                     
                obj.calculateFEC;
            else
                % disp('ROI object is invalid or deleted.');
            end
        end        
        
        % adjust fur/eye threshold for binarized image
        function adjustThreshold(obj, ~, ~)          
            obj.threshold = obj.sliderThreshold.Value;
            txtThreshold.String = ['Threshold: ', num2str(obj.threshold, '%.0f')];
            if isfield(obj, 'roiHandle') && isvalid(obj.roiHandle)
                obj.updateBinaryVideo;
            end
            disp(txtThreshold);
        end


        function onGUIClose(obj, ~, ~)

            close(obj.EBC_vid_log_trial);

            obj.stopGUIVideo(); % Ensure the live video is stopped
            obj.saveSettings(); % Save settings before closing
            delete(obj.hFig); % Close the figure
            % delete(timerfindall)
        end
        
        % calculate fraction eye closure
        function calculateFEC(obj)
            totalEllipsePixels = numel(find(obj.mask == 1));            
            openPixels =  sum(obj.binFrame(:) == 1); % Assuming open eye corresponds to 1, but it's 0
            obj.fec = (openPixels / totalEllipsePixels) * 100; % FEC percentage
            currentTime = seconds(datetime('now') - obj.startTime);
            obj.fecData = [obj.fecData, obj.fec];
            obj.fecTimes = [obj.fecTimes, currentTime];
            set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);
        end

        % calculate reference for 'eye open'
        function maxOpen = calculateMaxOpenness(obj, videoObj)
            while videoObj.CurrentTime < 60 % First 60 seconds
                obj.frame = readFrame(videoObj);
                binFrame = imbinarize(rgb2gray(obj.frame), graythresh(rgb2gray(obj.frame)));
                openPixels = sum(binFrame(:));
                maxOpen = max(maxOpen, openPixels);
            end
        end
        
        function saveSettings(obj)
            obj.guiSettings.thresholdValue = get(obj.sliderThreshold, 'Value');
            obj.guiSettings.eyeOpenThreshold = get(obj.sliderEyeOpen, 'Value');
            % save('guiSettings.mat', 'obj.guiSettings');
            disp('GUI settings saved.');
        end

        % Function to start handling trials
        function startTrial(obj)
            disp('Trial Started');
            recordVideoDuringTrial();
            saveVideoData(); % Save video after trial (during ITI)
        end

        % Function to record video during trial
        function recordVideoDuringTrial()
            startTime = ledOnset - 1; % Start recording 1s before LED onset
            stopTime = puffOffset + 2; % Stop recording 2s after puff offset
            trialVideoFrames = {};
            video.CurrentTime = startTime;
           while video.CurrentTime <= stopTime && hasFrame(video)
               frame = readFrame(video);
                trialVideoFrames{end+1} = frame;
                imshow(frame, 'Parent', axVideo); % Display current frame
                drawnow;
           end
        end
        
        % Function to save video log data
        function saveVideoData(obj, subject, fn)
           % Save video log data to file
            videoFilename = ['TrialVideo_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.mat'];
            save(videoFilename, 'trialVideoFrames');
            disp(['Video saved as ', videoFilename]);
        end        

    end
end

function getFrameFromCamera(img)
    pause(0.1)
end



