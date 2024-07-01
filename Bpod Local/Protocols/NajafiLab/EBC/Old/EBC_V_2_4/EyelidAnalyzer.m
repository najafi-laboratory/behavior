classdef EyelidAnalyzer < handle
    properties   
        AirPuffOnsetTime

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

        camEvents
        currentFrame

        EBC_vid_log_trial
        eyeOpen
        
        fec
        fecData = [];
        fecDataHist = [];
        fecRaw;
        fecNorm;
        fecPlot
        FECStartSlider;
        fecTimes = [];
        fecTimesHist = [];
        fpsCheck = 0;
        fps
        frame
        
        hFig

        isPaused = true;
        isPlaying = false;
        imgOrigHandle;
        imgBinHandle;
        isZoom = false;

        LEDOnsetTime
        
        maxOpen = 0;
        mask
        minFur = 0;
        
        roiHandle
        roiPosition

        binarizationThresholdSlider
        sliderEyeOpen
        src
        subjectName

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
        trialStartTime
        trialVidStartTime

        vid        
        video
        % vidWriter
        vidTimer

        startTime  % To store the start time of the video
        stopTime      

        syncLog
    end
    
    methods
        function obj = EyelidAnalyzer(subjectName)
            % Init Properties

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
            obj.binarizationThresholdSlider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 255, 'Value', 100, 'Units', 'normalized',...
                'Position', [0.64,0.51,0.18,0.03], 'Callback', @obj.adjustBinarizationThreshold);
            obj.FECStartSlider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', 0.7, 'Units', 'normalized',...
                'Position', [0.72,0.47,0.10,0.03], 'Callback', @obj.adjustFECStartThreshold);
            obj.tAdjustThresh = uicontrol('Style', 'text', 'Position', [423,337.5,171.5,20], ...
                'String', 'Adjust FEC Binarization Threshold');
            obj.tMinThresh = uicontrol('Style', 'text', 'Position', [489.5,310.5,191.5,20], ...
                'String', 'Adjust FEC Start Threshold');
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
                     
            % Load GUI settings if exist
            if exist('guiSettings.mat', 'file')
                load('guiSettings.mat', 'guiSettings');
                obj.guiSettings = guiSettings;
                % Set the GUI components to last saved states
                % set(obj.binarizationThresholdSlider, 'Value', obj.guiSettings.thresholdValue);
                % set(obj.sliderEyeOpen, 'Value', obj.guiSettings.eyeOpenThreshold);
            end

            obj.imgOrigHandle = imshow(obj.frame, 'Parent', obj.axOriginal);
            obj.imgBinHandle =  imshow(obj.frame, 'Parent', obj.axThresholded);

            obj.startPreTrialVideo;

        end

        function onGUIClose(obj, ~, ~)
            % close(obj.EBC_vid_log_trial);
            delete(obj.vid)
            clear obj.vid

            obj.stopPreTrialVideo(); % Ensure the live video is stopped
            % obj.saveSettings(); % Save settings before closing
            delete(obj.hFig); % Close the figure
            % delete(timerfindall)
        end     

        function stopPreTrialVideo(obj, ~, ~)
            if ~isempty(obj.vidTimer) && isvalid(obj.vidTimer)
                stop(obj.vidTimer);
                obj.vidTimer = [];
            end
            
            if ~isempty(obj.vid) && isvalid(obj.vid)
                stop(obj.vid);
            end
        end

        function startPreTrialVideo(obj, ~, ~)
            if isempty(obj.vid) || ~isvalid(obj.vid)
                % obj.vid = videoinput('pointgrey', 1, "F7_Raw8_640x512_Mode1");
                obj.vid = videoinput('gentl', 1, "Mono8");
                obj.src = getselectedsource(obj.vid);
            else
                stop(obj.vid);
            end
            
            % obj.src.ExposureMode = "Auto";
            obj.src.TriggerDelay = 0;
            % obj.src.FrameRateMode = "Manual";
            % obj.src.FrameRate = 473.6355;
            obj.src.AcquisitionFrameRate = 473.6355;
            % obj.src.Shutter = 0.01;
            % obj.src.Strobe3 = 'On';
            triggerconfig(obj.vid, "manual");
            obj.vid.FramesPerTrigger = inf;
            obj.vid.TriggerRepeat = 0;
            
            obj.src.LineSelector = 'Line2';
            obj.src.LineMode = 'Input';

            obj.camEvents = obj.vid.EventLog;
            
            obj.vid.LoggingMode = "memory";

            start(obj.vid);
            trigger(obj.vid);
            obj.startTime = datetime("now");            

            % Create and start a timer to update the video display
            if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
                obj.vidTimer = timer('TimerFcn',@(x,y)obj.updatePreTrialVideo(), 'ExecutionMode', 'fixedSpacing', 'Period', 0.030);
                start(obj.vidTimer);
                disp(['Video Update Timer Started']);
            else
                start(obj.vidTimer);
                disp(['Video Update Timer reStarted']);
            end            
        end

        function updatePreTrialVideo(obj, ~, ~)
           if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before getting images 
            end

            if obj.vid.FramesAvailable > 0
                % disp([' FramesAvailable ' num2str(obj.vid.FramesAvailable)])
                % [data, time, metadata] = getdata(obj.vid, obj.vid.FramesAvailable);
                % obj.frame = data(:,:,:,end);
                obj.frame = getsnapshot(obj.vid);
                rgbFrame = double(cat(3, obj.frame, obj.frame, obj.frame)) / 255; % Convert to RGB by replicating the single channel, Normalize to [0, 1] double precision
                set(obj.imgOrigHandle, 'CData', rgbFrame);
                flushdata(obj.vid);
                obj.updateBinaryVideo;   
                obj.calculateFEC('pretrial');
            end
        end

        function startTrialsVideo(obj, currentTrial, subjectName)
            obj.camEvents = obj.vid.EventLog;
            obj.vid.FramesPerTrigger = 2350;
            % obj.vid.LoggingMode = "disk&memory";
            obj.vid.LoggingMode = "memory";
            % obj.EBC_vid_log_trial = VideoWriter([obj.trialVideoDir, subjectName, '_TrialVid_', num2str(currentTrial), '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.avi'], 'Grayscale AVI');
            obj.EBC_vid_log_trial = VideoWriter([obj.trialVideoDir, subjectName, '_TrialVid_', num2str(currentTrial), '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.avi'], 'MPEG-4');
            % obj.EBC_vid_log_trial.FrameRate = obj.src.FrameRate; % err for framerate too high
            % obj.vid.DiskLogger = obj.EBC_vid_log_trial;
            open(obj.EBC_vid_log_trial);

            obj.clearFecPlot();

            start(obj.vid);
            % trigger(obj.vid);
            % obj.trialVidStartTime = datetime("now");

            % Create a timer to update the video display and get fec data
            if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
                obj.vidTimer = timer('TimerFcn',@(x,y)obj.updateTrialsVideo(), 'ExecutionMode', 'fixedSpacing', 'Period', 0.030, 'StartDelay',0.005);
                % start(obj.vidTimer);
                disp(['Video Update Timer Created']);
            else
                % start(obj.vidTimer);
            end  

            obj.syncLog = struct;
            obj.syncLog.time = [];
            obj.syncLog.lineStatus = [];
        end

        function triggerTrialsVideo(obj)
            trigger(obj.vid);
            obj.trialVidStartTime = datetime("now");
            start(obj.vidTimer);
            
        end

        function processTrialsVideo(obj)

            stop(obj.vid);

            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before getting images 
            end

            if obj.vid.FramesAvailable > 0
                % disp([' FramesAvailable ' num2str(obj.vid.FramesAvailable)])
                numFrames = obj.vid.FramesAvailable;
                [data, time, metadata] = getdata(obj.vid, numFrames);
                % obj.frame = data(:,:,:,end);
                checkFps = 0;
                if checkFps == 1
                    timeCheck = time - time(1);
                    Ts = diff(time);
                    avgFps_MeanBased = 1/mean(Ts)
                    avgFps_NumFramesPerVidTime = length(time)/time(end)
                end
                % rgbFrame = double(cat(3, obj.frame, obj.frame, obj.frame)) / 255; % Convert to RGB by replicating the single channel, Normalize to [0, 1] double precision
                % set(obj.imgOrigHandle, 'CData', rgbFrame);

                % update to use parallel pool for writing video data
                writeVideo(obj.EBC_vid_log_trial, data);
                test = milliseconds(obj.AirPuffOnsetTime - obj.LEDOnsetTime)
                flushdata(obj.vid);
                obj.setTrialData();
                obj.clearFecPlot();
                for i = 1:numFrames
                    obj.frame = data(:,:,:,i);
                    obj.binarizeImage();
                    % obj.updateBinaryVideo();
                    obj.calculatePostFEC(time(i));
                    % pause(0.0002);
                end
                % figure();
                % montage(data);
                pause(1);
                % obj.updateBinaryVideo;   
                % obj.calculateFEC;
            end            

            close(obj.EBC_vid_log_trial);

            if ~isempty(obj.vidTimer) && isvalid(obj.vidTimer)
                stop(obj.vidTimer);
            end

            % while (obj.vid.FramesAcquired ~= obj.vid.DiskLoggerFrameCount)   
            %     disp(['FramesAcquired ' num2str(obj.vid.FramesAcquired) '  DiskLoggerFrameCount ' num2str(obj.vid.DiskLoggerFrameCount)])
            %     pause(0.1)
            % end     
        end        

        function stopTrialsVideo(obj)
            if ~isempty(obj.vidTimer) && isvalid(obj.vidTimer)
                stop(obj.vidTimer);
                delete(obj.vidTimer);
                obj.vidTimer = [];
            end
            
            if ~isempty(obj.vid) && isvalid(obj.vid)
                stop(obj.vid);
            end            
        end

        function updateTrialsVideo(obj)
            % obj.syncLog.time = [obj.syncLog.time; second(datetime('now'))];
            % obj.syncLog.lineStatus = [obj.syncLog.lineStatus; obj.src.LineStatus];

            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before getting images 
            end

            if obj.vid.FramesAvailable > 0
                % disp([' FramesAvailable ' num2str(obj.vid.FramesAvailable)])
                % [data, time, metadata] = getdata(obj.vid, obj.vid.FramesAvailable);
                % obj.frame = data(:,:,:,end);
                obj.frame = getsnapshot(obj.vid);

                rgbFrame = double(cat(3, obj.frame, obj.frame, obj.frame)) / 255; % Convert to RGB by replicating the single channel, Normalize to [0, 1] double precision
                set(obj.imgOrigHandle, 'CData', rgbFrame);

                % writeVideo(obj.EBC_vid_log_trial, data);

                % flushdata(obj.vid);
                obj.updateBinaryVideo;   
                obj.calculateFEC('trial');
            end
        end

        % add other session/trial vars here to set at trial start
        function setTrialData(obj)
            obj.fecData = [];
            obj.fecTimes = [];
        end

        function plotLEDOnset(obj)
            % set(gca, 'NextPlot', 'add');  % Set the NextPlot property to 'add' to add graphics objects without clearing the existing plot
            % line(obj.fecPlot, [obj.LEDOnsetTime obj.LEDOnsetTime], ylim, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);  % Red dashed vertical lines at each x_bars(i)
            % set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);
        end

        function plotAirPuffOnset(obj)
            
        end

        function binarizeImage(obj)
            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
                obj.mask = createMask(obj.roiHandle); % Create a mask from the ROI
                grayFrame = im2gray(obj.frame);
                grayFrame(~obj.mask) = 0; % Apply mask to the frame
                obj.binFrame = imbinarize(grayFrame, obj.threshold / 255);
            end
        end

        function updateBinaryVideo(obj)
            obj.binarizeImage();
            set(obj.imgBinHandle, 'CData', obj.binFrame);
        end

        function calculateFEC(obj, mode)
            totalEllipsePixels = numel(find(obj.mask == 1));  % Total area inside the ellipse
            eyeAreaPixels = sum(obj.binFrame(obj.mask == 1) == 0);  % Black pixels inside the ellipse
            
            furPixels = totalEllipsePixels - eyeAreaPixels;
            obj.minFur = min(obj.minFur, furPixels);
                        
            fec = 1 - (eyeAreaPixels / (totalEllipsePixels-obj.minFur));
            
            set(obj.tFECVal, 'String', num2str(fec, '%.3f'));

            obj.fec = fec * 100;  % Convert to percentage
            switch mode
                case 'pretrial'
                    currentTime = seconds(datetime('now') - obj.startTime);
                case 'trial'
                    currentTime = seconds(datetime('now') - obj.trialVidStartTime);
            end
            

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

            set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);
            % set(obj.fecPlot, 'XData', obj.fecTimesHist, 'YData', obj.fecDataHist);
        end

        function calculatePostFEC(obj, currentTime)
            totalEllipsePixels = numel(find(obj.mask == 1));  % Total area inside the ellipse
            eyeAreaPixels = sum(obj.binFrame(obj.mask == 1) == 0);  % Black pixels inside the ellipse
            
            furPixels = totalEllipsePixels - eyeAreaPixels;
            obj.minFur = min(obj.minFur, furPixels);
                        
            fec = 1 - (eyeAreaPixels / (totalEllipsePixels-obj.minFur));
            
            set(obj.tFECVal, 'String', num2str(fec, '%.3f'));

            obj.fec = fec * 100;  % Convert to percentage
            % switch mode
            %     case 'pretrial'
            %         currentTime = seconds(datetime('now') - obj.startTime);
            %     case 'trial'
            %         currentTime = seconds(datetime('now') - obj.trialVidStartTime);
            % end

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

            set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);
            % set(obj.fecPlot, 'XData', obj.fecTimesHist, 'YData', obj.fecDataHist);
        end

        function clearFecPlot(obj)
            set(obj.fecPlot, 'XData', [], 'YData', []);
        end

        function selectROI(obj, ~, ~)
            % obj.pauseVideo;
            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                % position before getting images 
                delete(obj.roiHandle);
            end
            obj.roiHandle = imellipse(obj.axOriginal); % Let user draw an ellipse
            obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before playing  
            % obj.pauseVideo;
        end

        function adjustBinarizationThreshold(obj, ~, ~)
            obj.threshold = obj.binarizationThresholdSlider.Value;
            set(obj.tFECThreshVal, 'String', num2str(obj.threshold, '%.3f'));
            obj.updateBinaryVideo;
        end

        function adjustFECStartThreshold(obj, ~, ~)
            

            % also draw horiz line on plot

        end

        function saveSettings(obj)
            obj.guiSettings.thresholdValue = get(obj.binarizationThresholdSlider, 'Value');
            obj.guiSettings.eyeOpenThreshold = get(obj.FECStartSlider, 'Value');
            % save('guiSettings.mat', 'obj.guiSettings');
            disp('GUI settings saved.');
        end        


    end
end