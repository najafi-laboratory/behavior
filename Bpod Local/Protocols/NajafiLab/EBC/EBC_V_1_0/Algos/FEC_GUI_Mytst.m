% Run the GUI setup function
createGUI();

function createGUI()
    % Create the main figure
    hFig = figure('Name', 'Mouse Eye Video Analysis', 'NumberTitle', 'off', 'Resize', 'on');

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

    % Add controls and buttons
    sliderThreshold = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', 0.5, ...
              'Position', [150 50 120 20], 'Callback', @adjustThreshold);
    sliderEyeOpen = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', 0.7, ...
              'Position', [150 30 120 20], 'Callback', @adjustEyeOpenThreshold);
    uicontrol('Style', 'text', 'Position', [20 50 120 20], 'String', 'Adjust Threshold:');
    uicontrol('Style', 'text', 'Position', [20 30 200 20], 'String', 'Minimum Eye Openness Threshold:');
    uicontrol('Style', 'pushbutton', 'String', 'Start', ...
              'Position', [300 20 50 20], 'Callback', @startVideo);
    uicontrol('Style', 'pushbutton', 'String', 'Stop', ...
              'Position', [360 20 50 20], 'Callback', @stopVideo);
    uicontrol('Style', 'pushbutton', 'String', 'Load Video & Draw ROI', ...
              'Position', [420 20 150 20], 'Callback', @loadVideoAndDrawROI);

    % Global variables
    global video eyeOpenMax thresholdEyeOpen fecData fecTimes binaryVideos guiSettings vid vidTimer;

    % Initialize data storage
    fecData = [];
    fecTimes = [];
    binaryVideos = {};

    % Load GUI settings if exist
    if exist('guiSettings.mat', 'file')
        load('guiSettings.mat', 'guiSettings');
        % Set the GUI components to last saved states
        set(sliderThreshold, 'Value', guiSettings.thresholdValue);
        set(sliderEyeOpen, 'Value', guiSettings.eyeOpenThreshold);
    end

% Function to start live video
    function startVideo(src, event)
        % Find available hardware
        info = imaqhwinfo;

        % Connect to a Video Input Camera
        vid = videoinput('gentl', 1);

        % Configure Video Source Properties
        src = getselectedsource(vid);

        % Set GPIO Line3 as Output (if needed)
        % src.LineSelector = "Line3";
        % src.LineMode = "Output";

        % Configure the logging mode to disk
        % vid.LoggingMode = 'disk';
        % EBC_vid_log = VideoWriter('EBC_vid.mp4', 'MPEG-4');
        % vid.DiskLogger = EBC_vid_log;

        % Start the acquisition
        start(vid);

        vidTimer = [];
        % Create and start a timer to update the video display
        if isempty(vidTimer) || ~isvalid(vidTimer)
            vidTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.1, 'TimerFcn', @updateVideo);
            start(vidTimer);
        end
    end


    % Function to stop live video
    function stopVideo(src, event)
        if ~isempty(vidTimer) && isvalid(vidTimer)
            stop(vidTimer);
            delete(vidTimer);
            vidTimer = [];
        end

        if ~isempty(vid) && isvalid(vid)
            stop(vid);
            delete(vid);
        end

        imaqreset; % Reset Image Acquisition Toolbox
        clear vid; % Clear the video object
    end

    % Function to update video display
    function updateVideo(~, ~)
        if vid.FramesAvailable > 0
            frame = getsnapshot(vid); % Capture the current frame
            imshow(frame, 'Parent', axOriginal); % Display the frame in the original video axes
        end
    end

% Function to be called when GUI is closed or session ends
    function onGUIClose(~, ~)
        stopLiveVideo(); % Ensure the live video is stopped
        saveSettings(); % Save settings before closing
        finalizeSession(); % Optional, depends on your application logic
        delete(hFig); % Close the figure
    end

    % Function to start handling trials
    function startTrial()
        video.CurrentTime = 0; % Reset video to trial start
        trialDuration = 1; % seconds after puff offset
        currentTrialData = [];
        currentTrialTimes = [];

        while hasFrame(video) && video.CurrentTime <= trialDuration
            frame = readFrame(video);
            currentTime = video.CurrentTime;
            binFrame = imbinarize(rgb2gray(frame), graythresh(rgb2gray(frame)));
            imshow(frame, 'Parent', axOriginal); % Display original frame
            imshow(binFrame, 'Parent', axThresholded); % Display binary frame

            % Store binary frame for video saving
            binaryVideos{end+1} = binFrame;

            % Calculate FEC
            currentFEC = calculateFEC(binFrame);
            currentTrialData = [currentTrialData, currentFEC];
            currentTrialTimes = [currentTrialTimes, currentTime];

            % Update FEC plot
            set(fecPlot, 'XData', currentTrialTimes, 'YData', currentTrialData);
            drawnow; % Update plot in real-time
        end

        % Store trial data
        fecData = [fecData; currentTrialData];
        fecTimes = [fecTimes; currentTrialTimes];
        saveTrialData(binaryVideos, currentTrialData, currentTrialTimes); % Save data
    end

    % Function to calculate Fraction of Eyelid Closure
    function fec = calculateFEC(binaryImage)
        totalPixels = numel(binaryImage);
        closedPixels = sum(binaryImage(:) == 0); % Assuming closed eye corresponds to 0
        fec = (closedPixels / totalPixels) * 100; % FEC percentage
    end

    % Function to save trial data
    function saveTrialData(binaryVideoFrames, fecData, fecTimes)
        save('binaryVideoFrames.mat', 'binaryVideoFrames');
        save('fecData.mat', 'fecData');
        save('fecTimes.mat', 'fecTimes');
        disp('Trial data saved.');
    end


    % Callback function to save GUI settings
    function saveSettings()
        guiSettings.thresholdValue = get(sliderThreshold, 'Value');
        guiSettings.eyeOpenThreshold = get(sliderEyeOpen, 'Value');
        save('guiSettings.mat', 'guiSettings');
        disp('GUI settings saved.');
    end

    % % Function to initialize session
    % function initializeSession()
    %     % Assuming video initialization for a pre-recorded video
    %     video = VideoReader('sampleVideo.mp4');
    %     eyeOpenMax = calculateMaxOpenness(video);
    %     thresholdEyeOpen = get(sliderEyeOpen, 'Value'); % Initial threshold
    % end

    % Function to calculate max openness from the initial video frames
    function maxOpen = calculateMaxOpenness(videoObj)
        maxOpen = 0; % Initialize maximum openness
        videoObj.CurrentTime = 0;
        while videoObj.CurrentTime < 60 % First 60 seconds
            frame = readFrame(videoObj);
            binFrame = imbinarize(rgb2gray(frame), graythresh(rgb2gray(frame)));
            openPixels = sum(binFrame(:));
            maxOpen = max(maxOpen, openPixels);
        end
    end

    % Set the GUI close request function
    set(hFig, 'CloseRequestFcn', @onGUIClose);

end

