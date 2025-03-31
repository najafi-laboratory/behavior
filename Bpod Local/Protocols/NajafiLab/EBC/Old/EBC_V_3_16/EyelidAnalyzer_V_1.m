% Grouping Properties

classdef EyelidAnalyzer_V_1 < handle
    properties
        % GUI Components
        hFig
        axOriginal
        axThresholded
        axFEC
        sliderThreshold
        sliderEyeOpen
        bstartVideo
        bstopVideo
        bselectROI
        tAdjustThresh
        tMinThresh
        fecPlot

        % Video Properties
        vid
        src
        vidTimer
        frame
        binFrame
        binaryVideos
        mask
        roiHandle
        roiPosition
        fpsCheck = 0
        fps
        video
        isPaused = true
        isPlaying = false
        isZoom = false

        % Analysis Data
        fecData
        fecTimes
        threshold = 100
        thresholdEyeOpen
        maxOpen = 0
        eyeOpenMax
        fec

        % Settings
        guiSettings

        % Logging
        EBC_vid_log_trial
        trialVideoDir = '.\Videos\TrialVideos\'

        % Timing
        startTime
        stopTime
    end
   

methods

% Constructor and GUI Methods
    function obj = EyelidAnalyzer_V_1()
        % Constructor: Initialize properties and create GUI
        obj.fecData = [];
        obj.fecTimes = [];
        obj.binaryVideos = {};
        obj.createGUI();
    end

    function createGUI(obj)

        % Create the main GUI window and components
        obj.hFig = figure('Name', 'Mouse Eye Video Analysis', 'Position', [12 324 961 666], 'NumberTitle', 'off', 'Resize', 'on', 'CloseRequestFcn', @obj.onGUIClose);

        obj.axOriginal = subplot(2, 2, 1, 'Parent', obj.hFig);
        title(obj.axOriginal, 'Original Video');
        obj.axThresholded = subplot(2, 2, 2, 'Parent', obj.hFig);
        title(obj.axThresholded, 'Thresholded Video');
        obj.axFEC = subplot(2, 1, 2, 'Parent', obj.hFig);
        title(obj.axFEC, 'Fraction of Eyelid Closure Over Time');
        xlabel(obj.axFEC, 'Time (s)');
        ylabel(obj.axFEC, 'FEC (%)');
        hold(obj.axFEC, 'on');
        obj.fecPlot = plot(obj.axFEC, NaN, NaN, 'b-');

        obj.sliderThreshold = uicontrol('Style', 'slider', 'Min', 0, 'Max', 255, 'Value', 100, 'Position', [150 50 120 20], 'Callback', @obj.adjustThreshold);
        obj.sliderEyeOpen = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', 0.7, 'Position', [150 30 120 20], 'Callback', @obj.adjustEyeOpenThreshold);
        obj.tAdjustThresh = uicontrol('Style', 'text', 'Position', [20 50 120 20], 'String', 'Adjust Threshold:');
        obj.tMinThresh = uicontrol('Style', 'text', 'Position', [20 30 200 20], 'String', 'Minimum Eye Openness Threshold:');
        obj.bstartVideo = uicontrol('Style', 'pushbutton', 'String', 'Start', 'Position', [300 20 50 20], 'Callback', @obj.startGUIVideo);
        obj.bstopVideo = uicontrol('Style', 'pushbutton', 'String', 'Stop', 'Position', [360 20 50 20], 'Callback', @obj.stopGUIVideo);
        obj.bselectROI = uicontrol('Style', 'pushbutton', 'String', 'Select ROI', 'Units', 'normalized', 'Position', [0.25, 0.25, 0.15, 0.05], 'Callback', @obj.selectROI);
        uicontrol('Parent', obj.hFig, 'Style', 'pushbutton', 'String', 'Pause', 'Units', 'normalized', 'Position', [0.25, 0.15, 0.15, 0.05], 'Callback', @obj.pauseVideo);

        if exist('guiSettings.mat', 'file')
            load('guiSettings.mat', 'guiSettings');
            obj.guiSettings = guiSettings;
            set(obj.sliderThreshold, 'Value', obj.guiSettings.thresholdValue);
            set(obj.sliderEyeOpen, 'Value', obj.guiSettings.eyeOpenThreshold);
        end
    end

% Video Control Methods
    function connectVideo(obj)
        obj.vid = videoinput('gentl', 1);
        obj.src = getselectedsource(obj.vid);
        obj.fps = obj.src.AcquisitionFrameRate;

        if obj.fpsCheck && ~((obj.fps > 30) && (obj.fps < 31))
            obj.onGUIClose();
            error('Set camera FPS to 30 fps. fps = %s', num2str(obj.fps));
        end

        obj.src.LineSelector = 'Line2';
        obj.src.LineMode = 'Input';
        triggerconfig(obj.vid, 'hardware');
        obj.src.TriggerMode = 'On';
        obj.src.TriggerActivation = 'RisingEdge';
        obj.vid.FramesPerTrigger = 200;
        obj.src.LineSelector = 'Line3';
        obj.src.LineMode = 'Output';

        warning('off', 'imaq:getdata:infFramesPerTrigger');
    end

    function disconnectVideo(obj)
        if ~isempty(obj.vid) && isvalid(obj.vid)
            delete(obj.vid);
        end

        imaqreset;
        clear obj.vid;
    end

    function startGUIVideo(obj, ~, ~)
        obj.connectVideo();
        start(obj.vid);

        while ~isrunning(obj.vid)
            pause(0.1);
        end

        obj.isPaused = false;
        obj.startTime = datetime('now');

        if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
            obj.vidTimer = timer('TimerFcn', @(x, y) obj.updateVideo(), 'ExecutionMode', 'fixedRate', 'Period', 1);
            start(obj.vidTimer);
        end
    end

    function stopGUIVideo(obj, ~, ~)
        if ~isempty(obj.vidTimer) && isvalid(obj.vidTimer)
            stop(obj.vidTimer);
            delete(obj.vidTimer);
            obj.vidTimer = [];
        end

        if is_empty(obj.vid) && isvalid(obj.vid)
            stop(obj.vid);
            delete(obj.vid);
        end

        imaqreset;
        clear obj.vid;
    end

    function pauseVideo(obj, ~, ~)
        if obj.isPaused
            start(obj.vidTimer);
        else
            stop(obj.vidTimer);
        end
        obj.isPaused = ~obj.isPaused;
    end

% ROI Methods
    function selectROI(obj, ~, ~)
        obj.pauseVideo();
        if isempty(obj.vid)
            msgbox('Start video first!');
            return;
        end

        obj.roiHandle = imellipse(obj.axOriginal);
        obj.roiPosition = getPosition(obj.roiHandle);
        obj.pauseVideo();
    end

% Video Processing Methods
    function updateVideo(obj, ~, ~)
        if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
            obj.roiPosition = getPosition(obj.roiHandle);
        end

        if isrunning(obj.vid)
            [data, ~, ~] = getdata(obj.vid);
            obj.frame = data(:,:,:,end);
            if obj.isZoom && ~isempty(obj.frame)
                img_crop = obj.frame(409:550, 575:730);
                obj.frame = img_crop;
            end
            imshow(obj.frame, 'Parent', obj.axOriginal);
            flushdata(obj.vid);
        end

        if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
            setPosition(obj.roiHandle, obj.roiPosition);
        elseif ~isempty(obj.roiPosition)
            obj.roiHandle = imellipse(obj.axOriginal, obj.roiPosition);
        end

        obj.updateBinaryVideo();
    end

    function updateBinaryVideo(obj)
        if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
            obj.mask = createMask(obj.roiHandle);
            grayFrame = im2gray(obj.frame);
            grayFrame(~obj.mask) = 0;
            obj.binFrame = imbinarize(grayFrame, obj.threshold / 255);
            imshow(obj.binFrame, 'Parent', obj.axThresholded);
            obj.calculateFEC();
        end
    end

    function adjustThreshold(obj, ~, ~)
        obj.threshold = obj.sliderThreshold.Value;
        if isvalid(obj.roiHandle)
            obj.updateBinaryVideo();
        end
    end

 % Analysis Methods
    function calculateFEC(obj)
        totalEllipsePixels = numel(find(obj.mask == 1));
        openPixels = sum(obj.binFrame(:) == 1);
        obj.fec = (openPixels / totalEllipsePixels) * 100;
        currentTime = seconds(datetime('now') - obj.startTime);
        obj.fecData = [obj.fecData, obj.fec];
        obj.fecTimes = [obj.fecTimes, currentTime];
        set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);
    end

 % Save Settings and GUI Close Methods
    function saveSettings(obj)
        obj.guiSettings.thresholdValue = get(obj.sliderThreshold, 'Value');
        obj.guiSettings.eyeOpenThreshold = get(obj.sliderEyeOpen, 'Value');
        save('guiSettings.mat', 'obj.guiSettings');
    end

    function onGUIClose(obj, ~, ~)
        obj.stopGUIVideo();
        obj.saveSettings();
        delete(obj.hFig);
    end
end

end





