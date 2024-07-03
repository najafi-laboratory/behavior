%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

function eyeAnalysisApp

    % Create the main figure
    hFig = figure('Name', 'Mouse Eye Analysis', 'NumberTitle', 'off', 'MenuBar', 'none', 'Toolbar', 'none', 'Resize', 'on');


    % Axes for displaying videos
    axOriginal = axes('Parent', hFig, 'Units', 'normalized', 'Position', [0.05, 0.35, 0.4, 0.6]);
    axBinary = axes('Parent', hFig, 'Units', 'normalized', 'Position', [0.55, 0.35, 0.4, 0.6]);


    % UI controls
    btnLoad = uicontrol('Parent', hFig, 'Style', 'pushbutton', 'String', 'Load Video', 'Units', 'normalized', 'Position', [0.05, 0.25, 0.15, 0.05], 'Callback', @loadVideo);
    btnSelectROI = uicontrol('Parent', hFig, 'Style', 'pushbutton', 'String', 'Select ROI', 'Units', 'normalized', 'Position', [0.25, 0.25, 0.15, 0.05], 'Callback', @selectROI);
    btnPlay = uicontrol('Parent', hFig, 'Style', 'pushbutton', 'String', 'Play', 'Units', 'normalized', 'Position', [0.05, 0.15, 0.15, 0.05], 'Callback', @playVideo);
    sliderThreshold = uicontrol('Parent', hFig, 'Style', 'slider', 'Units', 'normalized', 'Position', [0.55, 0.25, 0.3, 0.05], 'Min', 0, 'Max', 255, 'Value', 100, 'Callback', @adjustThreshold);
    txtThreshold = uicontrol('Parent', hFig, 'Style', 'text', 'String', 'Threshold: 100', 'Units', 'normalized', 'Position', [0.45, 0.25, 0.1, 0.05]);


    % Variables to store video and ROI data
    videoData = struct('videoObject', [], 'currentFrame', [], 'roiHandle', [], 'threshold', 100, 'isPlaying', false);


    % Function to load video

    function loadVideo(~, ~)
        [file, path] = uigetfile({'*.avi;*.mp4', 'Video Files (*.avi, *.mp4)'});
        if isequal(file, 0)
            return;
        end
        videoData.videoObject = VideoReader(fullfile(path, file));
        videoData.currentFrame = readFrame(videoData.videoObject);
        imshow(videoData.currentFrame, 'Parent', axOriginal);
        if isobject(videoData.roiHandle) && isvalid(videoData.roiHandle)
            delete(videoData.roiHandle); % Remove previous ROI if it exists
        end
    end


    % Function to select ROI

    function selectROI(~, ~)
        if isempty(videoData.videoObject)
            msgbox('Load a video first!');
            return;
        end
        videoData.roiHandle = imellipse(axOriginal); % Let user draw an ellipse
        wait(videoData.roiHandle); % Optional: wait for user to finalize the ellipse
    end


    % Function to play video

      function playVideo(~, ~)
        if isempty(videoData.videoObject) || isempty(videoData.roiHandle) || ~isvalid(videoData.roiHandle)
            msgbox('Load a video and select an ROI first!');
            return;
        end
        videoData.isPlaying = true;
        roiPosition = getPosition(videoData.roiHandle);  % Get ROI position before playing
        while hasFrame(videoData.videoObject) && videoData.isPlaying
            videoData.currentFrame = readFrame(videoData.videoObject);
            imshow(videoData.currentFrame, 'Parent', axOriginal);
            if isobject(videoData.roiHandle) && isvalid(videoData.roiHandle)
                setPosition(videoData.roiHandle, roiPosition);  % Reset the position for consistency
            else
                videoData.roiHandle = imellipse(axOriginal, roiPosition);  % Recreate ROI if not valid
            end
            updateBinaryVideo();
            drawnow;
        end
      end


    % Function to adjust threshold

    function adjustThreshold(src, ~)
        videoData.threshold = src.Value;
        txtThreshold.String = ['Threshold: ', num2str(videoData.threshold, '%.0f')];
        if isfield(videoData, 'roiHandle') && isvalid(videoData.roiHandle)
            updateBinaryVideo();
        end
    end

   
    % Function to update binary video

    function updateBinaryVideo
        if isempty(videoData.currentFrame)
            return;
        end
        if isobject(videoData.roiHandle) && isvalid(videoData.roiHandle)
            mask = createMask(videoData.roiHandle); % Create a mask from the ROI
            grayFrame = rgb2gray(videoData.currentFrame);
            grayFrame(~mask) = 0; % Apply mask to the frame
            binaryFrame = imbinarize(grayFrame, videoData.threshold / 255);
            imshow(binaryFrame, 'Parent', axBinary);
        else
            disp('ROI object is invalid or deleted.');
        end

    

    end

end