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

        EBC_vid_log
        eyeOpenMax
        
        fec
        fecData        
        fecPlot
        fecTimes
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

        vid        
        video
        vidTimer

        startTime  % To store the start time of the video
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
            
            % Add controls and buttons
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
            obj.bloadVideoAndDrawROI = uicontrol('Style', 'pushbutton', 'String', 'Load Video & Draw ROI', ...
                'Position', [420 20 150 20], 'Callback', @obj.loadVideoAndDrawROI);
            
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

        function pauseVideo(obj, ~, ~)             
            if obj.isPaused
                start(obj.vidTimer)
            else                
                stop(obj.vidTimer);
            end
            obj.isPaused = ~obj.isPaused; % Toggle pause state
        end

        function startVideo(obj, ~, ~)
            % Find available hardware
            info = imaqhwinfo;
            
            % Connect to a Video Input Camera
            obj.vid = videoinput('gentl', 1);
            
            % Configure Video Source Properties
            obj.src = getselectedsource(obj.vid);
            
            % Configure the logging mode to disk
            % obj.vid.LoggingMode = 'disk';
            % obj.EBC_vid_log = VideoWriter('EBC_vid.mp4', 'MPEG-4');
            % obj.vid.DiskLogger = obj.EBC_vid_log;
            
            % Set trigger config
            obj.vid.FramesPerTrigger = inf;

            % Set video input timeout
            % set(obj.vid,'Timeout',720); %set the Timeout property of VIDEOINPUT object 'vid' to 50 seconds

            % Start the acquisition
            start(obj.vid);

            pause(.1);
            warning('off','imaq:getdata:infFramesPerTrigger')

            disp(['Video Acquisition Started']);

            obj.isPaused = false;        

            % Record the start time
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
                disp(['Video Input Buffer Empty']);
            end

            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
                setPosition(obj.roiHandle, obj.roiPosition);  % Reset the position for consistency
            elseif ~isempty(obj.roiPosition)
                obj.roiHandle = imellipse(obj.axOriginal, obj.roiPosition);  % Recreate ROI if not valid
            end

            obj.updateBinaryVideo;
        end

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
        
        function adjustThreshold(obj, ~, ~)          
            obj.threshold = obj.sliderThreshold.Value;
            txtThreshold.String = ['Threshold: ', num2str(obj.threshold, '%.0f')];
            if isfield(obj, 'roiHandle') && isvalid(obj.roiHandle)
                obj.updateBinaryVideo;
            end
            disp(txtThreshold);
        end

        function onGUIClose(obj, ~, ~)
            obj.stopVideo(); % Ensure the live video is stopped
            obj.saveSettings(); % Save settings before closing
            delete(obj.hFig); % Close the figure
            % delete(timerfindall)
        end
        
        function calculateFEC(obj)
            totalEllipsePixels = numel(find(obj.mask == 1));            
            openPixels =  sum(obj.binFrame(:) == 1); % Assuming open eye corresponds to 1, but it's 0
            obj.fec = (openPixels / totalEllipsePixels) * 100; % FEC percentage
            currentTime = seconds(datetime('now') - obj.startTime);
            obj.fecData = [obj.fecData, obj.fec];
            obj.fecTimes = [obj.fecTimes, currentTime];
            set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);
        end

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
    end
end
