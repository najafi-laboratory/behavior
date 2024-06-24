function createLiveVideoGUI()
    % Create the main GUI figure
    % hFig = figure('Name', 'Live Video Feed', 'NumberTitle', 'off', 'Position', [100, 100, 640, 480]);
    % Create an axes object to display the video
    % ax = axes('Parent', hFig, 'Units', 'pixels', 'Position', [10, 50, 620, 420]);
    % Initialize the webcam (assuming the first available webcam)
    % cam = webcam;
    % Create Start and Stop buttons
    % uicontrol('Parent', hFig, 'Style', 'pushbutton', 'String', 'Start Video', ...
    %           'Position', [10, 10, 100, 30], 'Callback', @startVideo);
    % uicontrol('Parent', hFig, 'Style', 'pushbutton', 'String', 'Stop Video', ...
              'Position', [120, 10, 100, 30], 'Callback', @stopVideo);
    % Variable to keep track of the timer object
    vidTimer = [];
    % Callback function to start video
    function startVideo(~, ~)
        if isempty(vidTimer) || ~isvalid(vidTimer)
            % Create a timer object to grab frames
            vidTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.1, 'TimerFcn', @updateVideo);
            start(vidTimer);  % Start the timer
        end
    end
    % Callback function to stop video
    function stopVideo(~, ~)
        if ~isempty(vidTimer) && isvalid(vidTimer)
            stop(vidTimer);  % Stop the timer
            delete(vidTimer);  % Delete the timer object
            vidTimer = [];
        end
    end
    % Timer callback function to update video in axes
    function updateVideo(~, ~)
        frame = snapshot(cam);  % Capture the current frame
        image(ax, frame);  % Display the frame
        ax.Visible = 'off';  % Turn off axes visibility
    end
    % Clean up on GUI close
    function closeGUI(~, ~)
        stopVideo();
        clear cam;  % Release the webcam
        delete(hFig);  % Close the figure
    end
    % Set the close request function of the figure
    set(hFig, 'CloseRequestFcn', @closeGUI);
end
% Call the GUI setup function
createLiveVideoGUI();