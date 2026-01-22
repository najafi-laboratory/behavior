classdef EyelidAnalyzer_PostProcessing_V_5_3 < handle
    properties   
        AirPuffOnsetTime
        AirPuffLine

        arrTotalEllipsePixels = []; 
        arrEyeAreaPixels = [];
        arrFECTrialStartThresh = [];        

        axOriginal
        axThresholded
        axFEC

        baselineSlider
        binFrame
        binaryVideos
        blinkThreshold
        bloadVideoAndDrawROI
        bselectROI
        bsetEyeOpenMax
        bstartVideo
        bstopVideo

        camEvents
        currentFrame

        EBC_vid_log_trial
        eyeAreaPixels            
        eyeOpen = false;
        eyeOpenAvgWindow = 0.1;  % 100ms
        
        fec
        fecAVG = 100;
        fecData = [];

        fecDataRaw = [];
        fecNorm;
        fecPlot
        FECStartSlider;
        FECStartThreshLine
        FECTrialStartThresh = 0.4;
        FECTrialStartThreshPercent;
        fecTimes = [];
        fecTimesHist = [];
        fpsCheck = 0;
        fps
        frame
        frameIndex
        FullFECRaw
        FullFEC 
        FullEyeAreaPixels
        
        hasSSImg = false;
        hFig

        isPaused = true;
        isPlaying = false;
        imgOrigHandle;
        imgBinHandle;
        ITI_pre;
        isZoom = false;

        LEDLine
        LEDOnsetTime
        
        maxOpen = 0;
        mask
        minFur = 0;
        
        nTrials
        
        roiHandle
        roiPosition

        ProtoCamPeriod = 0.004;

        binarizationThresholdSlider
        sliderEyeOpen
        src
        subjectName
        SessionDataDir
        SessionData
        SessionDataPath = []

        tAdjustThresh
        tCurrentTrialNumber_nonTimeout;
        tCurrentTrialNumber_nonTimeoutVal;
        tFEC
        tFECVal
        tFECThresh
        tFECThreshVal
        tFECTrialStartThresh;
        tFECTrialStartThreshVal;
        binarizationThreshold = 100;
        thresholdEyeOpen
        tMinThresh
        totalEllipsePixels
        trialVideoDir = 'C:\behavior\video_data\';
        trialStartTime
        trialVidStartTime
        trialNum
        
        vid        
        video
        % vidWriter
        vidTimer
        vidTimerPeriod = 0.030;
        vidNumFrames

        startTime  % To store the start time of the video
        stopTime      

        syncLog
    end
    
    methods
        function obj = EyelidAnalyzer_PostProcessing(subjectName)
            % Init Properties

            % Initialize data storage
            obj.fecData = [];
            obj.fecTimes = [];
            obj.binaryVideos = {};
            
            % Create GUI
            obj.createGUI();    

            obj.subjectName = subjectName;
              
            % obj.trialVideoDir = [obj.trialVideoDir, obj.subjectName, '\', datestr(now, 'yymmdd'), '\'];     
            % make video folder for current test subject if it doesn't already
            % exist
            [status, msg, msgID] = mkdir(obj.trialVideoDir); 

            obj.LoadFirstVideo;
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
            obj.FECStartSlider = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', obj.FECTrialStartThresh, 'Units', 'normalized',...
                'Position', [0.72,0.47,0.10,0.03], 'Callback', @obj.adjustFECStartThreshold);
           
            obj.tAdjustThresh = uicontrol('Style', 'text', 'Position', [423,337.5,171.5,20], ...
                'String', 'Adjust FEC Binarization Threshold');
            
            
            obj.tMinThresh = uicontrol('Style', 'text', 'Position', [489.5,310.5,191.5,20], ...
                'String', 'Adjust FEC Start Threshold');
           
            obj.tFECThresh = uicontrol('Style', 'text', 'Position', [801.5,337.5,42.5,20], ...
                'String', 'Thresh');
            obj.tFECThreshVal = uicontrol('Style', 'text', 'Position', [848.5,336.5,54.5,20], ...
                'String', '0.000');
            
            obj.tFECTrialStartThresh = uicontrol('Style', 'text', 'Position', [801.5,311.5,43.5,20], ...
                'String', 'Thresh');
            obj.tFECTrialStartThreshVal = uicontrol('Style', 'text', 'Position', [841.5,310.5,67.5,20], ...
                'String', '0.000');
            
            obj.tFEC = uicontrol('Style', 'text', 'Position', [798.5,10.5,33.5,22.5], ...
                'String', 'FEC');
            obj.tFECVal = uicontrol('Style', 'text', 'Position', [834.5,11.5,33.5,20], ...
                'String', '0.000');            

            obj.bselectROI = uicontrol('Style', 'pushbutton', 'String', 'Select ROI', 'Units', 'normalized',...
                'Position', [0.20,0.48,0.15,0.05], 'Callback', @obj.selectROI);
            obj.bsetEyeOpenMax = uicontrol('Style', 'pushbutton', 'String', 'Set Eye Open Max', 'Units', 'normalized',...
                'Position', [0.66,0.55,0.15,0.05], 'Callback', @obj.setEyeOpenMax);

            % obj.tCurrentTrialNumber_nonTimeout = uicontrol('Style', 'text', 'Position', [144.5,-1.5,154.5,20], ...
            %     'String', 'Trial Number non-timeout');
            % obj.tCurrentTrialNumber_nonTimeoutVal = uicontrol('Style', 'text', 'Position', [271.5,-2.5,67.5,20], ...
            %     'String', '0');            
            % obj.tFECTrialStartThresh = uicontrol('Style', 'text', 'Position', [801.5,311.5,43.5,20], ...
            %     'String', 'Trial Number non-timeout');
            % obj.tFECTrialStartThreshVal = uicontrol('Style', 'text', 'Position', [841.5,310.5,67.5,20], ...
            %     'String', '0');  


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

            obj.adjustFECStartThreshold;

            % obj.startPreTrialVideo;
            

            % L1 = addlistener(obj.axOriginal, 'XLim',  'PostSet', @(x,y)disp(y.AffectedObject.XLim));
            % L1 = addlistener(obj.axOriginal, 'XLim',  'PostSet', @obj.SetBinZoom);            
            % L2 = addlistener(obj.axOriginal, 'YLim',  'PostSet', @obj.SetBinZoom);

            % L1 = addlistener(obj.axOriginal, 'XAxis', 'LimitsChangedFcn', @obj.SetBinZoom);
            % L2 = addlistener(obj.axOriginal, 'YAxis', 'LimitsChangedFcn', @obj.SetBinZoom);            

            % obj.axOriginal.XAxis.LimitsChangedFcn = @(~,~)disp('LimitsChangedFcn');

            obj.axOriginal.XAxis.LimitsChangedFcn = @obj.SetBinZoom;
            obj.axOriginal.YAxis.LimitsChangedFcn = @obj.SetBinZoom;
        end

        function StopVideo(obj)
            if ~isempty(obj.vidTimer) && isvalid(obj.vidTimer)
                stop(obj.vidTimer);
                disp(['Video Update Timer Stopped']);
                obj.vidTimer = [];
            end
            
            % if ~isempty(obj.vid) && isvalid(obj.vid)
            %     stop(obj.vid);
            %     disp(['Video Update Timer Stopped']);
            % end            
        end

        function ProcessFEC(obj)
            obj.StopVideo();

            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                % obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before getting images 
                obj.roiPosition = obj.roiHandle.Position;
            end            
            
            obj.trialNum = 1;

            % Get a list of all .avi files in the specified directory
            fileList = dir(fullfile(obj.trialVideoDir, '*.avi'));

            fileNames = {fileList.name};
            sortedFileNames = sort(fileNames);            

            % load all trial videos to get max eye open
            % obj.nTrials
            % maxTrials = obj.nTrials + 1;
            maxTrials = obj.nTrials + 1;
            % maxTrials = 0;
            % checkVidTrials = 0;
            obj.FullEyeAreaPixels = [];
            while obj.trialNum < maxTrials
                obj.setTrialData()

                % Substring to match (e.g., 'video')
                substring = ['_TrialVid_', num2str(obj.trialNum), '_'];
                
                % Find the indices of the files containing the substring
                vidIndex = find(contains({fileList.name}, substring));
    
                videoFile = fileList(vidIndex).name;
                videoPath = [obj.trialVideoDir, videoFile];
    
                % Specify the path to your .avi file
                % videoFile = [obj.trialVideoDir, obj.subjectName, 'TrialVid_', ];  % Change this to your .avi file path
                
                % Create a VideoReader object to read the video
                obj.vid = VideoReader(videoPath);
                
                % Get information about the video (optional)
                disp(['Trial: ', num2str(obj.trialNum)]);
                disp(['Video: ', num2str(videoFile)]);
                disp(['Frame Rate: ', num2str(obj.vid.FrameRate)]);
                disp(['Number of Frames: ', num2str(obj.vid.NumFrames)]);
                disp(['Video Duration: ', num2str(obj.vid.Duration), ' seconds']); 
                disp(['-----------------------------------------']);
    
                obj.frameIndex = 1;      

                while hasFrame(obj.vid)
                    obj.frame = readFrame(obj.vid);
                    % rgbFrame = double(cat(3, obj.frame, obj.frame, obj.frame)) / 255; % Convert to RGB by replicating the single channel, Normalize to [0, 1] double precision
                    set(obj.imgOrigHandle, 'CData', obj.frame);
    
                    obj.updateBinaryVideo;   
                    obj.calculatePostFEC(obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECTimes(obj.frameIndex));
                    obj.frameIndex = obj.frameIndex +1;
                end

                obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECRaw = obj.fecDataRaw;
                % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FEC = obj.fecData;
                obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECTimes = obj.fecTimes;            
                obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.eyeAreaPixels = obj.arrEyeAreaPixels;
                obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.totalEllipsePixels = obj.arrTotalEllipsePixels;
              
                % obj.FullFECRaw = [obj.FullFECRaw, obj.fecDataRaw];
                % obj.FullFEC = [obj.FullFEC, obj.fecData];

                obj.FullEyeAreaPixels = [obj.FullEyeAreaPixels, obj.arrEyeAreaPixels];
             
                % pause(3);
                obj.trialNum = obj.trialNum + 1;                
            end

            maxEyeAreaPixels = max(obj.FullEyeAreaPixels); % max eye area 
            % obj.minFur = obj.totalEllipsePixels - maxEyeAreaPixels; % minimum fur
            % fec = 1 - (obj.eyeAreaPixels / (obj.totalEllipsePixels-obj.minFur)); 

            % process fec data using max eye open
            obj.trialNum = 1;
            while obj.trialNum < maxTrials
                test1 = obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FEC;
                test2 = 100 * (1 - (obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.eyeAreaPixels ./ maxEyeAreaPixels));
                % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECRaw = obj.fecDataRaw;
                obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FEC = 100*(1 - (obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.eyeAreaPixels ./ maxEyeAreaPixels));
                % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECTimes = obj.fecTimes;            
                % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.eyeAreaPixels = obj.arrEyeAreaPixels;
                % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.totalEllipsePixels = obj.arrTotalEllipsePixels;
                % 
                % obj.FullFECRaw = [obj.FullFECRaw, obj.fecDataRaw];
                % obj.FullFEC = [obj.FullFEC, obj.fecData];
                % 
                % obj.FullEyeAreaPixels = [obj.FullEyeAreaPixels, obj.arrEyeAreaPixels];
                
                set(obj.fecPlot, 'XData', obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECTimes, 'YData', obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FEC);
                
                % pause(1);
                obj.trialNum = obj.trialNum + 1;                
            end            
            % pause(3);

            SessionData =  obj.SessionData.SessionData;

            savePath = 'C:\behavior\session_data\E4L7\E4L7_EBC_V_3_12_20241104_143943_PP.mat';

            save(savePath, 'SessionData');

            obj.trialNum = 1;
            checkVidTrials = 5;
            while obj.trialNum < checkVidTrials
                obj.setTrialData()

                % Substring to match (e.g., 'video')
                substring = ['_TrialVid_', num2str(obj.trialNum), '_'];
                
                % Find the indices of the files containing the substring
                vidIndex = find(contains({fileList.name}, substring));
    
                videoFile = fileList(vidIndex).name;
                videoPath = [obj.trialVideoDir, videoFile];
    
                % Specify the path to your .avi file
                % videoFile = [obj.trialVideoDir, obj.subjectName, 'TrialVid_', ];  % Change this to your .avi file path
                
                % Create a VideoReader object to read the video
                obj.vid = VideoReader(videoPath);
                
                % Get information about the video (optional)
                disp(['Trial: ', num2str(obj.trialNum)]);
                disp(['Video: ', num2str(videoFile)]);
                disp(['Frame Rate: ', num2str(obj.vid.FrameRate)]);
                disp(['Number of Frames: ', num2str(obj.vid.NumFrames)]);
                disp(['Video Duration: ', num2str(obj.vid.Duration), ' seconds']); 
                disp(['-----------------------------------------']);
    
                obj.frameIndex = 1;      

                while hasFrame(obj.vid)
                    obj.frame = readFrame(obj.vid);
                    % rgbFrame = double(cat(3, obj.frame, obj.frame, obj.frame)) / 255; % Convert to RGB by replicating the single channel, Normalize to [0, 1] double precision
                    set(obj.imgOrigHandle, 'CData', obj.frame);
    
                    obj.updateBinaryVideo;   
                    % obj.calculatePostFEC('time');
                    set(obj.fecPlot, 'XData', obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECTimes(1:obj.frameIndex), 'YData', obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FEC(1:obj.frameIndex));
                    pause(0.004);
                    obj.frameIndex = obj.frameIndex +1;
                end
                % 
                % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECRaw = obj.fecDataRaw;
                % % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FEC = obj.fecData;
                % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECTimes = obj.fecTimes;            
                % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.eyeAreaPixels = obj.arrEyeAreaPixels;
                % obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.totalEllipsePixels = obj.arrTotalEllipsePixels;
              
                % obj.FullFECRaw = [obj.FullFECRaw, obj.fecDataRaw];
                % obj.FullFEC = [obj.FullFEC, obj.fecData];

                % obj.FullEyeAreaPixels = [obj.FullEyeAreaPixels, obj.arrEyeAreaPixels];
             
                % pause(3);
                obj.trialNum = obj.trialNum + 1;                
            end
            

        end

        function LoadFirstVideo(obj, x, y)

            obj.SessionDataPath = ['C:\behavior\session_data\E4L7\E4L7_EBC_V_3_12_20241104_143943.mat'];
            obj.SessionData = load(obj.SessionDataPath);
            
            dateStr = '241104';
            
            % obj.trialVideoDir = ['C:\behavior\video_data\', ;
            
            obj.trialVideoDir = [obj.trialVideoDir, obj.subjectName, '\', '241104', '\'];
            
            % Get a list of all .avi files in the specified directory
            fileList = dir(fullfile(obj.trialVideoDir, '*.avi'));

            fileNames = {fileList.name};
            sortedFileNames = sort(fileNames);

            obj.nTrials = obj.SessionData.SessionData.nTrials;
            obj.trialNum = 1;

            % Substring to match (e.g., 'video')
            substring = ['_TrialVid_', num2str(obj.trialNum), '_'];
            
            % Find the indices of the files containing the substring
            vidIndex = find(contains({fileList.name}, substring));

            videoFile = fileList(vidIndex).name;
            videoPath = [obj.trialVideoDir, videoFile];

            % Specify the path to your .avi file
            % videoFile = [obj.trialVideoDir, obj.subjectName, 'TrialVid_', ];  % Change this to your .avi file path
            
            % Create a VideoReader object to read the video
            obj.vid = VideoReader(videoPath);
            
            % Get information about the video (optional)
            disp(['Frame Rate: ', num2str(obj.vid.FrameRate)]);
            disp(['Number of Frames: ', num2str(obj.vid.NumFrames)]);
            disp(['Video Duration: ', num2str(obj.vid.Duration), ' seconds']); 

            obj.frameIndex = 1;

            % Create and start a timer to update the video display
            if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
                obj.vidTimer = timer('TimerFcn',@(x,y)obj.updatePostProcessVideo(), 'ExecutionMode', 'fixedSpacing', 'Period', 1/250);
                start(obj.vidTimer);
                disp(['Video Update Timer Started']);
            else
                start(obj.vidTimer);
                disp(['Video Update Timer reStarted']);
            end  

            

            % Check if any .avi files were found
            % if isempty(fileList)
            %     disp('No .avi files found in the specified directory.');
            % else
            %     % Display the list of .avi files found
            %     disp('List of .avi files found:');
            %     for k = 1:length(fileList)
            %         disp(fileList(k).name);
            %     end
            % end
            % 
            % obj.nTrials = SessionData.nTrials;
            % 
            % for trial = 1:obj.nTrials
            %     % filePath = [obj.trialVideoDir, obj.subjectName, '_', 
            %     % Check if the file exists
            %     if exist(filePath, 'file') == 2
            %         disp(['File exists: ', filePath]);
            %     else
            %         disp(['File does not exist: ', filePath]);
            %     end
            % end

            % obj.EBC_vid_log_trial = VideoWriter([obj.trialVideoDir, subjectName, '_TrialVid_', num2str(currentTrial), '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.avi'], 'Grayscale AVI');
           
        end

        function updatePostProcessVideo(obj, ~, ~)
           if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                % obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before getting images 
                obj.roiPosition = obj.roiHandle.Position;
            end

            
            if hasFrame(obj.vid)
                obj.frame = readFrame(obj.vid);
                % rgbFrame = double(cat(3, obj.frame, obj.frame, obj.frame)) / 255; % Convert to RGB by replicating the single channel, Normalize to [0, 1] double precision
                set(obj.imgOrigHandle, 'CData', obj.frame);

                obj.updateBinaryVideo;   
                obj.calculateFEC('pretrial', obj.SessionData.SessionData.RawEvents.Trial{1, obj.trialNum}.Data.FECTimes(obj.frameIndex)); 
                obj.frameIndex = obj.frameIndex +1;                
            else
                obj.frameIndex = 1;
                % Now, to restart from the beginning, reset the CurrentTime property
                % obj.vid.CurrentTime = 0;
                if obj.trialNum < obj.nTrials
                    obj.trialNum = obj.trialNum + 1;
                else
                    obj.trialNum = 0;
                end

                % Get a list of all .avi files in the specified directory
                fileList = dir(fullfile(obj.trialVideoDir, '*.avi'));
    
                fileNames = {fileList.name};
                sortedFileNames = sort(fileNames);

                % Substring to match (e.g., 'video')
                substring = ['_TrialVid_', num2str(obj.trialNum), '_'];
                
                % Find the indices of the files containing the substring
                vidIndex = find(contains({fileList.name}, substring));
    
                videoFile = fileList(vidIndex).name;
                videoPath = [obj.trialVideoDir, videoFile];
    
                % Specify the path to your .avi file
                % videoFile = [obj.trialVideoDir, obj.subjectName, 'TrialVid_', ];  % Change this to your .avi file path
                
                % Create a VideoReader object to read the video
                obj.vid = VideoReader(videoPath);
                
                % Get information about the video (optional)
                disp(['Frame Rate: ', num2str(obj.vid.FrameRate)]);
                disp(['Number of Frames: ', num2str(obj.vid.NumFrames)]);
                disp(['Video Duration: ', num2str(obj.vid.Duration), ' seconds']);                 
            end

            % obj.updateBinaryVideo;   
            % obj.calculateFEC('pretrial');
            % if obj.vid.FramesAvailable > 0
            %     % disp([' FramesAvailable ' num2str(obj.vid.FramesAvailable)])
            %     % [data, time, metadata] = getdata(obj.vid, obj.vid.FramesAvailable);
            %     % obj.frame = data(:,:,:,end);
            %     obj.frame = getsnapshot(obj.vid);
            %     rgbFrame = double(cat(3, obj.frame, obj.frame, obj.frame)) / 255; % Convert to RGB by replicating the single channel, Normalize to [0, 1] double precision
            %     set(obj.imgOrigHandle, 'CData', rgbFrame);
            %     flushdata(obj.vid);
            %     obj.updateBinaryVideo;   
            %     obj.calculateFEC('pretrial');
            % end
        end        


        function SetBinZoom(obj, x, y)
            % x = nargin
            % disp(y.AffectedObject.XLim);
            % 
            % obj.axThresholded.XLim = y.AffectedObject.XLim;
            % obj.axThresholded.YLim = y.AffectedObject.YLim;

            obj.axThresholded.XLim = obj.axOriginal.XLim;
            obj.axThresholded.YLim = obj.axOriginal.YLim;
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
            imaqreset;
            if isempty(obj.vid) || ~isvalid(obj.vid)
                obj.vid = videoinput('pointgrey', 1, "F7_Raw8_640x512_Mode1");
                % obj.vid = videoinput('gentl', 1, "Mono8");
                obj.src = getselectedsource(obj.vid);
            else
                stop(obj.vid);
            end
            
            % % obj.src.ExposureMode = "Auto";
            obj.src.TriggerDelay = 0;
            obj.src.FrameRateMode = "Manual";
            obj.src.FrameRate = 250;
            % obj.src.FrameRate = 473.6355;
            % obj.src.AcquisitionFrameRate = 473.6355;
            % obj.src.Shutter = 0.5;
            obj.src.Gain = 10;
            obj.src.Shutter = 3;
            obj.src.Strobe3 = 'On';
            triggerconfig(obj.vid, "manual");
            obj.vid.FramesPerTrigger = inf;
            obj.vid.TriggerRepeat = 0;
            
            % obj.src.LineSelector = 'Line2';
            % obj.src.LineMode = 'Input';

            obj.camEvents = obj.vid.EventLog;
            
            obj.vid.LoggingMode = "memory";

            obj.vid.ROIPosition = [160 142 448 370];

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
                % obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before getting images 
                obj.roiPosition = obj.roiHandle.Position;
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
            triggerCondition = "risingEdge";
            % triggerSource = "externalTriggerMode0-Source2";
            triggerSource = 'externalTriggerMode14-Source2';
            triggerconfig(obj.vid, "hardware", triggerCondition, triggerSource);

            obj.vid.ROIPosition = [160 142 448 370];
            
            obj.camEvents = obj.vid.EventLog;
            obj.vid.FramesPerTrigger = 1;
            obj.vid.TriggerRepeat = inf;
            % obj.vid.LoggingMode = "disk&memory";
            obj.vid.LoggingMode = "memory";
            % obj.EBC_vid_log_trial = VideoWriter([obj.trialVideoDir, subjectName, '_TrialVid_', num2str(currentTrial), '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.avi'], 'Grayscale AVI');            
            % obj.EBC_vid_log_trial = VideoWriter([obj.trialVideoDir, subjectName, '_TrialVid_', num2str(currentTrial), '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.avi'], 'Grayscale AVI');
            obj.EBC_vid_log_trial = VideoWriter([obj.trialVideoDir, subjectName, '_TrialVid_', num2str(currentTrial), '_', datestr(now, 'yyyy-mm-dd_HHMMSS'), '.avi'], 'Grayscale AVI');
            obj.EBC_vid_log_trial.FrameRate = 250; %400; % err for framerate too high
            % obj.EBC_vid_log_trial.Quality = 90;
            % obj.vid.DiskLogger = obj.EBC_vid_log_trial;
            open(obj.EBC_vid_log_trial);

            obj.clearFecPlot();            

            start(obj.vid);
            tic
            % trigger(obj.vid);
            % obj.trialVidStartTime = datetime("now");
            % obj.startTime = datetime('now');

            % Create a timer to update the video display and get fec data
            if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
                obj.vidTimer = timer('TimerFcn',@(x,y)obj.updateTrialsVideo(), 'ExecutionMode', 'fixedSpacing', 'Period', obj.vidTimerPeriod, 'StartDelay',0.005);
                % start(obj.vidTimer);
                disp(['Video Update Timer Created']);
            else
                % start(obj.vidTimer);
            end  

            % obj.syncLog = struct;
            % obj.syncLog.time = [];
            % obj.syncLog.lineStatus = [];

            % obj.trialVidStartTime = datetime("now");
            % start(obj.vidTimer);
            obj.setTrialData();
            obj.clearFecPlot();    
            obj.trialVidStartTime = datetime("now");
        end

        function triggerTrialsVideo(obj)
            % trigger(obj.vid);
            toc
            obj.setTrialData();
            obj.clearFecPlot();             
            obj.trialVidStartTime = datetime("now");
            start(obj.vidTimer);            
        end



        function checkEyeOpen(obj)         
            % obj.frame = getsnapshot(obj.vid);
            % rgbFrame = double(cat(3, obj.frame, obj.frame, obj.frame)) / 255; % Convert to RGB by replicating the single channel, Normalize to [0, 1] double precision
            % set(obj.imgOrigHandle, 'CData', rgbFrame);
            % obj.updateBinaryVideo;            
            % obj.calculateFEC('trial');
            
            % if (obj.fec / 100) <= obj.FECTrialStartThresh
            %     obj.eyeOpen = true;
            % else
            %     obj.eyeOpen = false;
            % end  

            % % get moving average window of FEC
            % numImgsToAvg = floor(obj.eyeOpenAvgWindow / obj.ProtoCamPeriod);            
            % if length(obj.fecData) >=  numImgsToAvg
            %     FECWindow = obj.fecData(end-numImgsToAvg+1:end);
            %     if sum(isnan(FECWindow)) == 0
            %         obj.fecAVG = mean(FECWindow);
            %     end
            % end

            if obj.hasSSImg
                if (obj.fecAVG / 100) <= obj.FECTrialStartThresh
                    obj.eyeOpen = true;
                else
                    obj.eyeOpen = false;
                end   
                obj.hasSSImg = false;
            end

            % if obj.vid.FramesAvailable > 0
            %     [data, time, metadata] = getdata(obj.vid, obj.vid.FramesAvailable);
            %     flushdata(obj.vid);
            % end
        end

        % function processTrialsVideo(obj, numFramesVid, numFramesITI, wasCheckEyeOpenTimeout)
        function processTrialsVideo(obj, numFramesVid, numFramesITI, numFramesVidKeep)
            if ~isempty(obj.vidTimer) && isvalid(obj.vidTimer)
                stop(obj.vidTimer);
            end

            stop(obj.vid);

            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)  
                % obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before getting images 
                obj.roiPosition = obj.roiHandle.Position;
            end

            % disp(['debug']);

            if obj.vid.FramesAvailable > 0
                % disp([' FramesAvailable ' num2str(obj.vid.FramesAvailable)])                
                % numFramesVid_ITI = min(obj.vid.FramesAvailable, numFramesVid + numFramesITI);
                numFramesVid = min(obj.vid.FramesAvailable, numFramesVid);
                numFramesPreVid = obj.vid.FramesAvailable - numFramesVid;
                if numFramesPreVid > 0
                    getdata(obj.vid, numFramesPreVid); % move intitial images, if any, to get to ITI_Pre seconds before LED_Onset
                end
                
                [data, time, metadata] = getdata(obj.vid, numFramesVidKeep);
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
                disp(['Writing Video File...']);
                tic
                % numFramesVid = numFramesVid;
                % numFramesVideo = numFramesVideo;
                writeVideo(obj.EBC_vid_log_trial, data(:,:,:,1:numFramesVidKeep));
                toc
                % test = milliseconds(obj.AirPuffOnsetTime - obj.LEDOnsetTime)
                flushdata(obj.vid);
                disp(['Processing FEC Data...']);
                tic
                obj.setTrialData()
                for i = 1:numFramesVidKeep
                    obj.frame = data(:,:,:,i);
                    obj.binarizeImage();
                    % obj.updateBinaryVideo();
                    obj.calculatePostFEC(time(i));
                    % pause(0.0002);
                end
                toc
                % figure();
                % montage(data);
                % pause(1);
                % obj.updateBinaryVideo;   
                % obj.calculateFEC;

                % obj.setTrialData();
                % obj.clearFecPlot();
                close(obj.EBC_vid_log_trial);
            else
                disp('no images? check proto');
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
                % obj.roiPosition = getPosition(obj.roiHandle);  % Get ROI position before getting images 
                obj.roiPosition = obj.roiHandle.Position;
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
                obj.hasSSImg = true;
            end
        end

        % add other session/trial vars here to set at trial start
        function setTrialData(obj)
            obj.eyeAreaPixels = 0;
            obj.fecDataRaw = [];
            obj.fecData = [];
            obj.fecTimes = [];
            obj.arrTotalEllipsePixels = []; 
            obj.arrEyeAreaPixels = [];
            obj.arrFECTrialStartThresh = [];   
            obj.fecAVG = 100;
        end

        function setEventTimes(obj, LEDOnsetTime, AirPuffOnsetTime, ITI_pre)
            obj.LEDOnsetTime = LEDOnsetTime;
            obj.AirPuffOnsetTime = AirPuffOnsetTime;
            obj.ITI_pre = ITI_pre;
        end

        function binarizeImage(obj)
            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
                obj.mask = createMask(obj.roiHandle); % Create a mask from the ROI
                grayFrame = im2gray(obj.frame);
                grayFrame(~obj.mask) = 0; % Apply mask to the frame
                obj.binFrame = imbinarize(grayFrame, obj.binarizationThreshold / 255);
            end
        end

        function updateBinaryVideo(obj)
            obj.binarizeImage();
            set(obj.imgBinHandle, 'CData', obj.binFrame);
        end

        function calculateFEC(obj, mode, currentTime)
            obj.totalEllipsePixels = numel(find(obj.mask == 1));  % Total area inside the ellipse
            obj.eyeAreaPixels = sum(obj.binFrame(obj.mask == 1) == 0);  % Black pixels inside the ellipse
            
            furPixels = obj.totalEllipsePixels - obj.eyeAreaPixels;
                        
            fec = 1 - (obj.eyeAreaPixels / (obj.totalEllipsePixels-obj.minFur));
            
            set(obj.tFECVal, 'String', num2str(fec, '%.3f'));

            obj.fec = fec * 100;  % Convert to percentage
            % switch mode
            %     case 'pretrial'
            %         currentTime = seconds(datetime('now') - obj.startTime);
            %     case 'trial'
            %         currentTime = seconds(datetime('now') - obj.trialVidStartTime);
            % end
            % currentTime = obj.vid.CurrentTime;
            
            obj.fecData = [obj.fecData, obj.fec];
            obj.fecTimes = [obj.fecTimes, currentTime];

            % get moving average window of FEC
            % numImgsToAvg = floor(obj.eyeOpenAvgWindow / obj.ProtoCamPeriod);  % if using logging fps          
            % numImgsToAvg = floor(obj.eyeOpenAvgWindow / obj.vidTimerPeriod);  % if using logging fps
            % if length(obj.fecData) >=  numImgsToAvg
            %     FECWindow = obj.fecData(end-numImgsToAvg+1:end);
            %     if sum(isnan(FECWindow)) == 0
            %         obj.fecAVG = mean(FECWindow);
            %     end
            % end

            avgWindowStart = currentTime - obj.eyeOpenAvgWindow;
            if avgWindowStart > 0
                avgWindowStartIdx = find(obj.fecTimes < avgWindowStart);
                if ~isempty(avgWindowStartIdx)
                    % check time difference to see if enough values for
                    % averaging window
                    % fecTimesAvgWindowLength = obj.fecTimes(averagingWindowIdxs(end)) - obj.fecTimes(averagingWindowIdxs(1));
                    % if fecTimesAvgWindowLength > obj.eyeOpenAvgWindow
                        FECWindow = obj.fecData(avgWindowStartIdx(end):end);
                        if sum(isnan(FECWindow)) == 0
                            % disp(num2str(obj.fecTimes(end) - obj.fecTimes(avgWindowStartIdx(end)));
                            % disp([' t2 - t1 = ' num2str(obj.fecTimes(end) - obj.fecTimes(avgWindowStartIdx(end)))]);
                            obj.fecAVG = mean(FECWindow);
                        end                        
                    % end

                end

            end

            set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);

            set(obj.FECStartThreshLine,'xdata', [0 currentTime], 'ydata', [obj.FECTrialStartThreshPercent  obj.FECTrialStartThreshPercent]);
        end

        function calculatePostFEC(obj, currentTime)
            obj.totalEllipsePixels = numel(find(obj.mask == 1));  % Total area inside the ellipse
            obj.eyeAreaPixels = sum(obj.binFrame(obj.mask == 1) == 0);  % Black pixels inside the ellipse
            
            furPixels = obj.totalEllipsePixels - obj.eyeAreaPixels;
                        
            obj.arrTotalEllipsePixels = [obj.arrTotalEllipsePixels obj.totalEllipsePixels];
            obj.arrEyeAreaPixels = [obj.arrEyeAreaPixels obj.eyeAreaPixels];            
            % obj.arrFECTrialStartThresh = [obj.arrFECTrialStartThresh obj.FECTrialStartThresh];            

            % calculate raw fec
            fec = 1 - (obj.eyeAreaPixels / (obj.totalEllipsePixels));

            % get raw fec data
            obj.fecDataRaw = [obj.fecDataRaw, fec];

            % calculate fec adjusted by max_eye_open_baseline
            fec = 1 - (obj.eyeAreaPixels / (obj.totalEllipsePixels-obj.minFur));            

            set(obj.tFECVal, 'String', num2str(fec, '%.3f'));

            obj.fec = fec * 100;  % Convert to percentage

            % currentTime = obj.vid.CurrentTime;
                     
            obj.fecData = [obj.fecData, obj.fec];
            obj.fecTimes = [obj.fecTimes, currentTime];
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
            obj.roiHandle =  drawfreehand(obj.axOriginal);            
            obj.roiPosition = obj.roiHandle.Position;
            obj.mask = createMask(obj.roiHandle);
        end

        % set max eye open to current number of white pixels as baseline
        function setEyeOpenMax(obj, ~, ~)
            if isobject(obj.roiHandle) && isvalid(obj.roiHandle)
                obj.totalEllipsePixels = numel(find(obj.mask == 1));  % Total area inside the ellipse
                obj.eyeAreaPixels = sum(obj.binFrame(obj.mask == 1) == 0);  % Black pixels inside the ellipse                
                obj.minFur = obj.totalEllipsePixels - obj.eyeAreaPixels;     
            end            
        end

        function adjustBinarizationThreshold(obj, ~, ~)
            obj.binarizationThreshold = obj.binarizationThresholdSlider.Value;
            set(obj.tFECThreshVal, 'String', num2str(obj.binarizationThreshold, '%.3f'));
            obj.updateBinaryVideo;
        end

        function initPlotLines(obj)
            obj.LEDLine = line(obj.axFEC, [0 0], ylim, 'Color', 'r', 'LineStyle', 'none', 'LineWidth', 2);
            obj.AirPuffLine = line(obj.axFEC, [0 0], ylim, 'Color', 'r', 'LineStyle', 'none', 'LineWidth', 2);
            fecStart = get(obj.FECStartSlider, 'Value');
            obj.FECStartThreshLine = line(obj.axFEC, xlim, [fecStart fecStart], ylim, 'Color', 'r', 'LineStyle', ':', 'LineWidth', 2);


            % line(obj.fecPlot, [obj.LEDOnsetTime obj.LEDOnsetTime], ylim, 'Color', 'r', 'LineStyle', ':', 'LineWidth', 2);
            % line(obj.fecPlot, [obj.AirPuffOnsetTime obj.AirPuffOnsetTime], ylim, 'Color', 'r', 'LineStyle', ':', 'LineWidth', 2);
            % line(obj.fecPlot, [obj.FECTrialStartThresh obj.FECTrialStartThresh], ylim, 'Color', 'r', 'LineStyle', ':', 'LineWidth', 2);            

        end

        function plotLEDOnset(obj)
            % set(gca, 'NextPlot', 'add');  % Set the NextPlot property to 'add' to add graphics objects without clearing the existing plot
            % line(obj.fecPlot, [obj.LEDOnsetTime obj.LEDOnsetTime], ylim, 'Color', 'r', 'LineStyle', ':', 'LineWidth', 2);  % Red dashed vertical lines at each x_bars(i)
            % set(obj.fecPlot, 'XData', obj.fecTimes, 'YData', obj.fecData);
            % LEDTime = obj.LEDOnsetTime + obj.ITI_pre;
            LEDTime = obj.LEDOnsetTime;
            set(obj.LEDLine,'xdata',[LEDTime LEDTime], 'ydata', [0 100], 'LineStyle', ':');
        end

        function plotAirPuffOnset(obj)
            % AirPuffTime = obj.trialVidStartTime + obj.AirPuffOnsetTime;
            % AirPuffTime = obj.AirPuffOnsetTime + obj.ITI_pre;
            AirPuffTime = obj.AirPuffOnsetTime;
            set(obj.AirPuffLine,'xdata',[AirPuffTime AirPuffTime], 'ydata', [0 100], 'LineStyle', ':');
        end

        function adjustFECStartThreshold(obj, ~, ~)
            obj.FECTrialStartThresh = get(obj.FECStartSlider, 'Value');
            obj.FECTrialStartThreshPercent = 100 * obj.FECTrialStartThresh;
            set(obj.tFECTrialStartThreshVal, 'String', num2str(obj.FECTrialStartThreshPercent, '%.3f'));
            set(obj.FECStartThreshLine,'ydata',[obj.FECTrialStartThreshPercent, obj.FECTrialStartThreshPercent]);
            % also draw horiz line on plot
        end

        function updateTrialNumber_nonTimeout(obj, currentTrialNumber_nonTimeout)            
            set(obj.tCurrentTrialNumber_nonTimeoutVal, 'String', num2str(currentTrialNumber_nonTimeout, '%.3f'));
        end

        function saveSettings(obj)
            obj.guiSettings.thresholdValue = get(obj.binarizationThresholdSlider, 'Value');
            obj.guiSettings.eyeOpenThreshold = get(obj.FECStartSlider, 'Value');
            % save('guiSettings.mat', 'obj.guiSettings');
            disp('GUI settings saved.');
        end        


    end
end