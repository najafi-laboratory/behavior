classdef EyelidAnalyzer < handle
    properties        
        axOriginal
        axThresholded
        axFEC

        baselineSlider
        binFrame
        binaryVideos
        blinkThreshold
        bloadVideoAndDrawROI
        bselectROI
        bstartVideo
        bstopVideo

        currentFrame

        EBC_vid_log_trial
        eyeOpen
        
        fec
        fecData = [];
        fecDataHist = [];
        fecRaw;
        fecNorm;
        fecPlot
        fecTimes = [];
        fecTimesHist = [];
        fpsCheck = 0;
        fps
        frame

        guiSettings
        
        hFig

        isPaused = true;
        isPlaying = false;
        imgOrigHandle;
        imgBinHandle;
        isZoom = false;

        maxOpen = 0;
        mask
        minFur = 0;
        
        roiHandle
        roiPosition

        sliderThreshold
        sliderEyeOpen
        src
        SubjectName

        tAdjustThresh
        tFEC
        tFECVal
        tFECThresh
        tFECThreshVal
        tFECBlink
        tFECBlinkVal
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
            % obj.SubjectName = SubjectName;

            % Initialize data storage
            obj.fecData = [];
            obj.fecTimes = [];
            obj.binaryVideos = {};
            
            % Create GUI
            obj.createGUI();    
        end
        
        function createGUI(obj)
            warning('off','imaq:getdata:infFramesPerTrigger');
            warning('off','imaq:gige:adaptorErrorPropSet');
            warning('off','MATLAB:JavaEDTAutoDelegation');
            warning('off','MATLAB:subplot:InvalidPositionSyntax');

            % console warning for java.awt.Component subclass method call automatically
            % delegated to the Event Dispatch Thread (EDT)
            % Create the main figure
            obj.hFig = figure('Name', 'Mouse Eye Video Analysis', 'Position', [12,324,961,666], 'NumberTitle', 'off', 'Resize', 'on', 'CloseRequestFcn', @obj.onGUIClose);
            
            % Create axes for displaying the original and thresholded videos
            obj.axOriginal = subplot(2, 2, 1, 'Parent', obj.hFig);
            title(obj.axOriginal, 'Original Video');
            obj.axThresholded = subplot(2, 2, 2, 'Parent', obj.hFig);
            title(obj.axThresholded, 'Thresholded Video');
            
            % Create a plot for Fraction of Eyelid Closure (FEC)
            obj.axFEC = subplot(2, 1, 2, 'Parent', obj.hFig, 'Position', [0.13,0.11,0.77,0.32]);
            title(obj.axFEC, 'Fraction of Eyelid Closure Over Time');
            xlabel(obj.axFEC, 'Time (s)');
            ylabel(obj.axFEC, 'FEC (%)');
            hold(obj.axFEC, 'on');
            obj.fecPlot = plot(obj.axFEC, NaN, NaN, 'b-'); % Initialize FEC plot
            
            % % Add controls and buttons
            obj.sliderThreshold = uicontrol('Style', 'slider', 'Min', 0, 'Max', 255, 'Value', 100, 'Units', 'normalized',...
                'Position', [0.64,0.51,0.18,0.03], 'Callback', @obj.adjustThreshold);
            obj.baselineSlider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', 0.7, 'Units', 'normalized',...
                'Position', [0.72,0.47,0.10,0.03], 'Callback', @obj.adjustFECBlinkThreshold);
            obj.tAdjustThresh = uicontrol('Style', 'text', 'Position', [423,337.5,171.5,20], ...
                'String', 'Adjust FEC Binarization Threshold');
            obj.tMinThresh = uicontrol('Style', 'text', 'Position', [489.5,310.5,191.5,20], ...
                'String', 'Adjust FEC Blink Threshold');
            obj.tFECThresh = uicontrol('Style', 'text', 'Position', [801.5,337.5,42.5,20], ...
                'String', 'Thresh');
            obj.tFECThreshVal = uicontrol('Style', 'text', 'Position', [848.5,336.5,54.5,20], ...
                'String', '0.000');
            obj.tFECBlink = uicontrol('Style', 'text', 'Position', [801.5,311.5,43.5,20], ...
                'String', 'Thresh');
            obj.tFECBlinkVal = uicontrol('Style', 'text', 'Position', [848.5,310.5,67.5,20], ...
                'String', '0.000');
            obj.tFEC = uicontrol('Style', 'text', 'Position', [798.5,10.5,33.5,22.5], ...
                'String', 'FEC');
            obj.tFECVal = uicontrol('Style', 'text', 'Position', [834.5,11.5,33.5,20], ...
                'String', '0.000');            
            % obj.bstartVideo = uicontrol('Style', 'pushbutton', 'String', 'Start', ...
            %     'Position', [300 20 50 20], 'Callback', @obj.startGUIVideo);
            % obj.bstopVideo = uicontrol('Style', 'pushbutton', 'String', 'Stop', ...
            %     'Position', [360 20 50 20], 'Callback', @obj.stopGUIVideo);
            % obj.bloadVideoAndDrawROI = uicontrol('Style', 'pushbutton', 'String', 'Load Video & Draw ROI', ...
            %     'Position', [420 20 150 20], 'Callback', @obj.loadVideoAndDrawROI);

            obj.bselectROI = uicontrol('Style', 'pushbutton', 'String', 'Select ROI', 'Units', 'normalized',...
                'Position', [0.20,0.48,0.15,0.05], 'Callback', @obj.selectROI);
            % uicontrol('Parent', obj.hFig, 'Style', 'pushbutton', 'String', 'Pause', ... 
            %     'Units', 'normalized', 'Position', [0.25, 0.15, 0.15, 0.05], 'Callback', @obj.pauseVideo);
                     
            obj.adjustThreshold;
            obj.adjustFECBlinkThreshold;
            % Load GUI settings if exist
            if exist('guiSettings.mat', 'file')
                load('guiSettings.mat', 'guiSettings');
                obj.guiSettings = guiSettings;
                % Set the GUI components to last saved states
                % set(obj.sliderThreshold, 'Value', obj.guiSettings.thresholdValue);
                % set(obj.sliderEyeOpen, 'Value', obj.guiSettings.eyeOpenThreshold);
            end

            %warning('on','verbose')
            % warning('off','imaq:getdata:infFramesPerTrigger');
            % warning('off','imaq:gige:adaptorErrorPropSet');
            % warning('off','MATLAB:JavaEDTAutoDelegation');
            % warning('off','MATLAB:subplot:InvalidPositionSyntax');
        end

        % connect video input
        function connectVideoTrial(obj, SubjectName)            
            % set strobe using pointgrey
            % v = videoinput("pointgrey", 1, "F7_Mono8_1280x1024_Mode0");
            % v.ROIPosition = [0 0 1280 1024];
            obj.vid = videoinput('pointgrey', 1, "F7_Raw8_640x512_Mode1");
            obj.src = getselectedsource(obj.vid);
            % obj.src.Exposure = 1.624512e+00;
            obj.src.ExposureMode = "Auto";
            % obj.src.FrameRate = 3.019109e+01;
            % obj.src.Sharpness = 512;
            % obj.src.SharpnessMode = "Auto";
            %obj.src.FrameRateMode = "Auto";
            obj.src.TriggerDelay = 1.13249 * 0.000001;
            obj.src.FrameRateMode = "Auto";
            obj.src.FrameRate = 473.6355;
            obj.src.Shutter = 2.00;
            obj.src.Strobe3 = 'On';
            % delete(obj.vid);


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
            framesPerTrigger = 500;
            % numTriggers = 200;
            numTriggers = 1;
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

            obj.trialVideoDir = ['C:\behavior\video_data\' obj.SubjectName '\' datestr(now, 'yymmdd') '\'];
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

            obj.src.ExposureMode = "Auto";
            % obj.src.SharpnessMode = "Auto";
            obj.src.FrameRateMode = "Auto";
            obj.src.FrameRate = 473.6355;
            obj.src.Shutter

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
            
%             obj.frame = getsnapshot(obj.vid);
% imshow(obj.frame, 'Parent', obj.axOriginal);

            % obj.imgOrigHandle = imshow(zeros(1024,1280,3), 'Parent', obj.axOriginal);
            % obj.imgBinHandle =  imshow(zeros(1024,1280,3), 'Parent', obj.axThresholded);

            obj.imgOrigHandle = imshow(obj.frame, 'Parent', obj.axOriginal);
            obj.imgBinHandle =  imshow(obj.frame, 'Parent', obj.axThresholded);


            % Create and start a timer to update the video display
            if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
                obj.vidTimer = timer('TimerFcn',@(x,y)obj.updateVideo(), 'ExecutionMode', 'fixedSpacing', 'Period', 0.065);
                % set(obj.vidTimer, 'TimerFcn', @(x,y)obj.updateVideo());
                start(obj.vidTimer);
                disp(['Video Update Timer Started']);
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
            % if isempty(obj.vid)
            %     msgbox('Start video first!');
            %     return;
            % end

            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                % obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI
                % position before getting images 
                delete(obj.roiHandle);
            end

            obj.roiHandle = imellipse(obj.axOriginal); % Let user draw an ellipse
            obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before playing  
            % disp(['ROI Selected']);
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

        function checkEyeOpenness(obj)
            while obj.vid.FramesAvailable > 0
                if obj.vid.FramesAvailable == 500
                    obj.vid.FramesAvailable
                    % disp(['FramesAvailable', num2str(obj.vid.FramesAvailable)]);
                    [data, time, metadata] = getdata(obj.vid, obj.vid.FramesAvailable);
    
                    timeCheck = time - time(1);
                    Ts = diff(time);
                    avgFps_MeanBased = 1/mean(Ts)
                    avgFps_NumFramesPerVidTime = length(time)/time(end)
    
                    timeCheck = 0;
                    if timeCheck
                        % check timing
                        timeCheck = time - time(1);
                        per = diff(time);
                        avgFps = 1/mean(per)
                        figure();
                        montage(data);
                    end
    
                    obj.frame = data(:,:,:,end);
                    rgbFrame = cat(3, obj.frame, obj.frame, obj.frame); % Convert to RGB by replicating the single channel
                    rgbFrame = double(rgbFrame) / 255; % Normalize to [0, 1] double precision
                    set(obj.imgOrigHandle, 'CData', rgbFrame);
                    obj.updateBinaryVideo;
                    obj.calculateFEC;
                     % Check baseline_fraction_eye_open threshold
                    baselineThreshold = (get(obj.baselineSlider, 'Value') * 100);
        
                    if obj.fec <= baselineThreshold
                        % Send FEC data to Bpod
                        % obj.sendFECtoBpod(obj.fec);
                        obj.eyeOpen = 1;
                        % Add these as while loop in proto 
                        disp('Trial started as eye is open more than the baseline threshold.');
                    else
                        obj.eyeOpen = 0;
                        disp('Eye is not open enough to start the trial.');
                    end
                    % obj.calculateFEC;
                    % end
                    % close(obj.EBC_vid_log_trial);
                    % obj.calculateFEC;
                end
            end            
            if ~isrunning(obj.vid)
                start(obj.vid);                    
            end
end

        function processVideoTrial(obj, currentTrial, SubjectName)
            % open(obj.EBC_vid_log_trial);  
            %disp(['Process Video']);
            while obj.vid.FramesAvailable > 0
                if (obj.vid.FramesPerTrigger == obj.vid.FramesAvailable)
                    obj.vid.FramesAvailable
                    % disp(['FramesAvailable', num2str(obj.vid.FramesAvailable)]);
                    [data, time, metadata] = getdata(obj.vid, obj.vid.FramesAvailable);
    
                    timeCheck = time - time(1);
                    Ts = diff(time);
                    avgFps_MeanBased = 1/mean(Ts)
                    avgFps_NumFramesPerVidTime = length(time)/time(end)
    
                    timeCheck = 0;
                    if timeCheck
                        % check timing
                        timeCheck = time - time(1);
                        per = diff(time);
                        avgFps = 1/mean(per)
                        figure();
                        montage(data);
                    end
    
                    obj.frame = data(:,:,:,end);
                    currentTrial = sprintf( '%03d', currentTrial);
                    % C:\behavior\video_data\SubjectName\062124
                    % [status, msg, msgID] = mkdir(obj.trialVideoDir);
                    % videoFilename = [[obj.trialVideoDir obj.SubjectName '\' datestr(now, 'yymmdd')], obj.SubjectName, '_TrialVid_', currentTrial, '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.avi'];
                    videoFilename = [obj.trialVideoDir, obj.SubjectName, '_TrialVid_', currentTrial, '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.avi'];
                    % videoFilename = [obj.trialVideoDir, SubjectName, '_TrialVid_', currentTrial, '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.avi'];
                    obj.EBC_vid_log_trial = VideoWriter(videoFilename, 'Grayscale AVI');                     
                    obj.EBC_vid_log_trial.FrameRate = avgFps_MeanBased;
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
                        rgbFrame = cat(3, obj.frame, obj.frame, obj.frame); % Convert to RGB by replicating the single channel
                        rgbFrame = double(rgbFrame) / 255; % Normalize to [0, 1] double precision
                        set(obj.imgOrigHandle, 'CData', rgbFrame);
                    end
                    close(obj.EBC_vid_log_trial);
                    obj.updateBinaryVideo;
                    obj.calculateFEC;
                end
            end
            if ~isrunning(obj.vid)
                start(obj.vid);
            end
            % flushdata(obj.vid);                   
        end    
     
        % update gui with most recent frame from image buffer
        function updateVideo(obj, ~, ~)     
            
            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before getting images 
            end

            if obj.vid.FramesAvailable > 0
                % obj.vid.FramesAvailable
                [data, time, metadata] = getdata(obj.vid);
                obj.frame = data(:,:,:,end);
                if obj.isZoom
                    if ~isempty(obj.frame)
                        img_crop = obj.frame(409:550, 575:730);
                        obj.frame = img_crop;
                        imshow(obj.frame, 'Parent', obj.axOriginal);
                    end
                else
                    rgbFrame = cat(3, obj.frame, obj.frame, obj.frame); % Convert to RGB by replicating the single channel
                    rgbFrame = double(rgbFrame) / 255; % Normalize to [0, 1] double precision
                    set(obj.imgOrigHandle, 'CData', rgbFrame);
                end
                flushdata(obj.vid);
                obj.updateBinaryVideo;   
                obj.calculateFEC;
            end
        end

        % update gui with roi-binarized image
        function updateBinaryVideo(obj)
            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
                obj.mask = createMask(obj.roiHandle); % Create a mask from the ROI
                grayFrame = im2gray(obj.frame);
                grayFrame(~obj.mask) = 0; % Apply mask to the frame
                obj.binFrame = imbinarize(grayFrame, obj.threshold / 255);
                set(obj.imgBinHandle, 'CData', obj.binFrame);                
                % obj.calculateFEC;

            end
        end        
        
        % adjust fur/eye threshold for binarized image
        function adjustThreshold(obj, ~, ~)          
            obj.threshold = obj.sliderThreshold.Value;
            % txtThreshold.String = ['Threshold: ', num2str(obj.threshold, '%.0f')];

            set(obj.tFECThreshVal, 'String', num2str(obj.threshold, '%.3f'));
            if isfield(obj, 'roiHandle') && isvalid(obj.roiHandle)
                obj.updateBinaryVideo;
            end
            % disp(txtThreshold);
        end


        function onGUIClose(obj, ~, ~)

            close(obj.EBC_vid_log_trial);

            obj.stopGUIVideo(); % Ensure the live video is stopped
            obj.saveSettings(); % Save settings before closing
            delete(obj.hFig); % Close the figure
            % delete(timerfindall)
        end
        
        % calculate fraction eye closure
        % function calculateFEC(obj)
        %     totalEllipsePixels = numel(find(obj.mask == 1));            
        %     openPixels =  sum(obj.binFrame(:) == 1); % Assuming open eye corresponds to 1, but it's 0
        %     obj.fec = (openPixels / totalEllipsePixels) * 100; % FEC percentage
        %     currentTime = seconds(datetime('now') - obj.startTime);
        %     obj.fecData = [obj.fecData, obj.fec];
        %     obj.fecTimes = [obj.fecTimes, currentTime];
        %     set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);
        % end
        function calculateFEC(obj)

            
            totalEllipsePixels = numel(find(obj.mask == 1));  % Total area inside the ellipse
            eyeAreaPixels = sum(obj.binFrame(obj.mask == 1) == 0);  % Black pixels inside the ellipse
            
            furPixels = totalEllipsePixels - eyeAreaPixels;
            obj.minFur = min(obj.minFur, furPixels);
            
            
            fec = 1 - (eyeAreaPixels / (totalEllipsePixels-obj.minFur));
            
            set(obj.tFECVal, 'String', num2str(fec, '%.3f'));


            obj.fec = fec * 100;  % Convert to percentage
            currentTime = seconds(datetime('now') - obj.startTime);

            % Append current FEC value to fecRaw array
            if isempty(obj.fecRaw)
                obj.fecRaw = obj.fec;  % Initialize if empty
            else
                obj.fecRaw = [obj.fecRaw, obj.fec];
            end

            obj.fecData = [obj.fecData, obj.fec];
            obj.fecTimes = [obj.fecTimes, currentTime];

            obj.fecDataHist = [obj.fecDataHist obj.fec];
            obj.fecTimesHist = [obj.fecTimesHist currentTime];

            % set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);
            set(obj.fecPlot, 'XData', obj.fecTimesHist, 'YData', obj.fecDataHist);
            


           
        end



        % calculate reference for 'eye open'

        function adjustFECBlinkThreshold(obj, ~, ~)
            obj.blinkThreshold = obj.baselineSlider.Value;
            set(obj.tFECBlinkVal, 'String', num2str(obj.blinkThreshold, '%.3f'));
        end
        
    function processFECData(obj)
            if isempty(obj.fecRaw)
                error('FEC raw data is empty');
            end
            
            % Calculate eye_area_max_open (minimum FEC value)
            eye_area_max_open = min(obj.fecRaw);
            
            % Calculate ROI as a reference area (maximum possible value of obj.fec)
            ROI = 1;  % This is just an assumption; change based on your actual ROI definition
            
            % Normalize the FEC values
            obj.fecNorm = ((ROI - obj.fecRaw) / (ROI - eye_area_max_open))*100;
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



