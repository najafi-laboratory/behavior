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

        EBC_vid_log_trial
        eyeOpenMax
        
        fec
        fecData        
        fecPlot
        fecTimes
        fpsCheck = 0;
        fps
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

            % start video input for ROI gui
            % obj.connectVideo;
            obj.startVideo;
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
                'Position', [300 20 50 20], 'Callback', @obj.startVideo);
            obj.bstopVideo = uicontrol('Style', 'pushbutton', 'String', 'Stop', ...
                'Position', [360 20 50 20], 'Callback', @obj.stopVideo);
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
        end

        % connect video input
        function connectVideo(obj, ~, ~)            
            % Connect to a Video Input Camera
            % vid = videoinput('winvideo', 2, 'YUY2_640x480');
            %imaqhwinfo

            obj.vid = videoinput('gentl', 1);     
            % obj.vid = videoinput('pointgrey', 1);
            disp(['Video Input Connected']);
            % Configure Video Source Properties
            obj.src = getselectedsource(obj.vid);

            % check fps of camera
            obj.fps = obj.src.AcquisitionFrameRate;

            obj.fpsCheck = 0;
            if obj.fpsCheck
                if ~((obj.fps > 30) && (obj.fps < 31))
                    %disp(['Set camera FPS = 30']);
                    obj.onGUIClose;
                    MatExc = MException('MyComponent:noSuchVariable','Set camera FPS to 30 fps. fps = %s', num2str(obj.fps));
                    throw(MatExc);               
                end
            end

            
           
            
            % Set trigger config
            % Configure the trigger type.
            % triggerconfig(obj.vid, 'manual')
            % obj.vid.FramesPerTrigger = inf;

            % obj.vid.TriggerType = 'hardware';
            obj.src.LineSelector = 'Line2';

                    % Configure the trigger settings.
            %triggerinfo(obj.vid)
            triggerconfig(obj.vid, 'hardware');
            % triggerconfig(obj.vid, 'hardware','risingEdge','TTL');
            obj.src.TriggerMode = 'On';
            obj.src.TriggerActivation = 'RisingEdge';
            obj.vid.FramesPerTrigger = 200;
            % obj.src.TriggerDelay = 0;

            % propinfo(obj.src,'LineSelector')
            % propinfo(obj.src,'LineSource')
            % propinfo(obj.src,'TriggerMode')
            % propinfo(obj.vid,'TriggerType')
            %obj.vid.FramesAvailable
            %obj.vid.FramesAcquired
            %isrunning(obj.vid)

            % Configure the logging mode to disk
            % obj.vid.LoggingMode = 'disk';
            % obj.EBC_vid_log_trial = VideoWriter('EBC_vid.mp4', 'MPEG-4');
            % obj.vid.DiskLogger = obj.EBC_vid_log_trial;            

            % trial video writer
            % make data folder for current test subject if it doesn't already
            % exist
% % Using character array
% folderName = 'Documents';
% subFolderName = 'MATLAB';
% directoryPath = fullfile('C:', 'Users', 'YourUsername', folderName, subFolderName);
% % Using string
% folderName = "Documents";
% subFolderName = "MATLAB";
% directoryPath = fullfile("C:", "Users", "YourUsername", folderName, subFolderName);

            obj.trialVideoDir = '.\Videos\TrialVideos\';
            [status, msg, msgID] = mkdir(obj.trialVideoDir);
            % videoFilename = [obj.trialVideoDir, 'Trial_', currentTrial, '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.mp4'];
            % obj.EBC_vid_log_trial = VideoWriter(videoFilename, 'MPEG-4');         
            % % obj.EBC_vid_log_trial = VideoWriter('EBC_vid.mp4', 'MPEG-4');            
            % open(obj.EBC_vid_log_trial);
            
            % Set video input timeout
            % set(obj.vid,'Timeout',720); %set the Timeout property of VIDEOINPUT object 'vid' to 50 seconds
            warning('off','imaq:getdata:infFramesPerTrigger')
        end

        % disconnect video input
        function disconnectVideo(obj, ~, ~)    
            if ~isempty(obj.vid) && isvalid(obj.vid)
                delete(obj.vid);
            end
            
            imaqreset; % Reset Image Acquisition Toolbox
            clear obj.vid; % Clear the video object
            disp(['Video Input Disconnected']);
        end  

        function startVideo(obj, ~, ~)          
            % Find available hardware
            % info = imaqhwinfo;
            
            % Connect to a Video Input Camera
            obj.vid = videoinput('gentl', 1);
            
            % Configure Video Source Properties
            obj.src = getselectedsource(obj.vid);

            % check fps of camera
            obj.fps = obj.src.AcquisitionFrameRate;
            if obj.fpsCheck
                if ~((obj.fps > 30) && (obj.fps < 31))
                    %disp(['Set camera FPS = 30']);
                    obj.onGUIClose;
                    MatExc = MException('MyComponent:noSuchVariable','Set camera FPS to 30 fps. fps = %s', num2str(obj.fps));
                    throw(MatExc);               
                end
            end
            %IMAQHELP(obj.src, 'AcquisitionFrameRate')
            % Determine the device specific frame rates (frames per second) available.
            % frameRates = set(obj.src, 'FrameRate')
            
            % Configure the logging mode to disk
            % obj.vid.LoggingMode = 'disk';
            % vid.LoggingMode = 'disk&memory';
            % obj.EBC_vid_log_trial = VideoWriter('EBC_vid.mp4', 'MPEG-4');
            % obj.vid.DiskLogger = obj.EBC_vid_log_trial;
            
            % Set trigger config
            obj.vid.FramesPerTrigger = inf;

            % Set video input timeout
            % set(obj.vid,'Timeout',720); %set the Timeout property of VIDEOINPUT object 'vid' to 50 seconds

            % Start the acquisition
            start(obj.vid);

            while ~isrunning(obj.vid)
                disp(['vid not started']);
            end

            pause(.1);
            warning('off','imaq:getdata:infFramesPerTrigger')

            disp(['Video Acquisition Started']);

            obj.isPaused = false;        

            % % Record the start time
            obj.startTime = datetime('now');
            
            % Create and start a timer to update the video display
            if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
                obj.vidTimer = timer('TimerFcn','', 'ExecutionMode', 'fixedRate', 'Period', 1);
                set(obj.vidTimer, 'TimerFcn', @(x,y)obj.updateVideo());
                start(obj.vidTimer);
                disp(['Video Update Timer Started']);
            end            
        end

        function stopVideo(obj, ~, ~)
            if ~isempty(obj.vidTimer) && isvalid(obj.vidTimer)
                stop(obj.vidTimer);
                delete(obj.vidTimer);
                obj.vidTimer = [];
            end
            
            if ~isempty(obj.vid) && isvalid(obj.vid)
                stop(obj.vid);
                delete(obj.vid);
            end
            
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

    
        % function triggerVideo(obj)
        %     Trigger the acquisition.
        %     trigger(obj.vid);
        % end
             
        function startVideoTrial(obj, ~, ~) 
            % obj.EBC_vid_log_trial = VideoWriter('EBC_vid.mp4', 'MPEG-4');
            % 
            % open(obj.EBC_vid_log_trial);
            % Start the acquisition
            % warning('off','imaq:getdata:infFramesPerTrigger')
            obj.isPaused = false;
            disp(['Video Trial Started']);
            start(obj.vid);
            
            % while obj.vid.FramesAvailable < 1
            %     disp(['vid not started']);
            % end

            % Record the start time
            % obj.startTime = datetime('now');
                        
            % Create and start a timer to update the video display
            % if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
            %     obj.vidTimer = timer('TimerFcn','', 'ExecutionMode', 'fixedRate', 'Period', 1);
            %     set(obj.vidTimer, 'TimerFcn', @(x,y)obj.updateVideo());
            %     start(obj.vidTimer);
            %     disp(['Video Update Timer Started']);
            % end            
        end       

        function stopVideoTrial(obj, ~, ~) 
            stop(obj.vid);
            obj.isPaused = true;
            % Record the stop time
            % obj.stopTime = datetime('now');
            % % vidTime = obj.startTime - obj.stopTime;
            % % vidTime
        end

        function processVideoTrial(obj, currentTrial)
            % open(obj.EBC_vid_log_trial);            
            while obj.vid.FramesAvailable > 0
                disp(['FramesAvailable', num2str(obj.vid.FramesAvailable)]);
                [data, time, metadata] = getdata(obj.vid, obj.vid.FramesAvailable);
                obj.frame = data(:,:,:,end);
                currentTrial = sprintf( '%03d', currentTrial);
                videoFilename = [obj.trialVideoDir, 'TrialVid_', datestr(now, 'yyyy-mm-dd_HHMMSS'), currentTrial, '.mp4'];
                obj.EBC_vid_log_trial = VideoWriter(videoFilename, 'MPEG-4');         
                % obj.EBC_vid_log_trial = VideoWriter('EBC_vid.mp4', 'MPEG-4');            
                open(obj.EBC_vid_log_trial);
                % update with binary vid?
                % writeVideo(obj.EBC_vid_log_trial, data);
                writeVideo(obj.EBC_vid_log_trial, data);
                %  montage(data)
                % update to add save video/log function(s)

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
            else
                % disp(['Video Input Buffer Empty']);
            end

            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
                setPosition(obj.roiHandle, obj.roiPosition);  % Reset the position for consistency
            elseif ~isempty(obj.roiPosition)
                obj.roiHandle = imellipse(obj.axOriginal, obj.roiPosition);  % Recreate ROI if not valid
            end

            obj.updateBinaryVideo;
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

            obj.stopVideo(); % Ensure the live video is stopped
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



