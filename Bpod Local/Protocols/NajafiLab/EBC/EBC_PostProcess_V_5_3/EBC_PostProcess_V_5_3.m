classdef EBC_PostProcess_V_5_3 < handle
    properties
        % Data/state
        SessionData = []
        sessionDataPath = ''
        sessionVideoPath = ''
        sessionVideoReader = []
        sessionVideoImage = []
        binarizedVideoImage = []
        currentFrame = []
        binFrame = []
        binarizationThreshold = 40
        roiHandle = []
        roiPosition = []
        roiMask = []
        videoDisplayScale = 0.5
        videoDuration = 0
        videoFrameRate = 0
        isPlaying = false
        playbackTimer = []
        fecDataRaw = []
        FEC = []
        fecTimes = []   % derived timestamp in seconds
        ellipsePixelSeries = []
        eyeAreaSeries = []
        minFur = []
        totalEllipsePixels = 0
        eyeAreaPixels = 0        
        secondCount = []
        cycleCount = []
        cycleOffset = []
        frameNibble = []    
        frameNumArr = [] % derived frame number array    
        timestampSec = []   % derived timestamp in seconds
        daqTrialSync = []
        daqCamStrobe = []
        daqTimestampsAll = []
        trialSyncRiseTimes = []
        camStrobeRiseTimes = []     
        droppedFrameNumbers = []
        camStrobeRiseTimesAdjusted = []      
        daqTimestampsAllSessionAligned = []    
        FECTimes = []               % exposure-centered timestamps

        % UI handles
        fig
        grid
        sessionLabel
        sessionEdit
        sessionBrowse
        videoGrid
        sessionVideoPanel
        sessionVideoPanelGrid
        sessionVideoAxes
        selectROIButton
        binarizedVideoPanel
        binarizedVideoPanelGrid
        binarizedVideoAxes
        fecThreshLabel
        fecThreshSlider
        findEyeOpenMaxButton
        leftPanel
        rightPanel
        buttonGrid
        loadDataButton
        loadVideoButton
        processButton
        exportButton
        videoControlsGrid
        playButton
        stopButton
        videoSlider        
    end

    methods
        function obj = EBC_PostProcess_V_5_3()
            warning('off','MATLAB:dispatcher:UnresolvedFunctionHandle');
            obj.buildUI();
        end

        function buildUI(obj)
            obj.fig = uifigure('Name','EBC Post Process','Position',[100 100 900 600]);

            obj.grid = uigridlayout(obj.fig,[5 3]);
            obj.grid.RowHeight = {'fit','2x','fit','1x','fit'};
            obj.grid.ColumnWidth = {'fit','fit','1x'};
            obj.grid.Padding = [15 15 15 15];
            obj.grid.RowSpacing = 10;
            obj.grid.ColumnSpacing = 10;

            obj.sessionLabel = uilabel(obj.grid,'Text','Session Folder:','HorizontalAlignment','right');
            obj.sessionLabel.Layout.Row = 1; obj.sessionLabel.Layout.Column = 1;

            obj.sessionBrowse = uibutton(obj.grid,'push','Text','Browse...', ...
                'ButtonPushedFcn',@obj.selectSessionFolder);
            obj.sessionBrowse.Layout.Row = 1; obj.sessionBrowse.Layout.Column = 2;

            obj.sessionEdit = uieditfield(obj.grid,'text');
            obj.sessionEdit.Layout.Row = 1; obj.sessionEdit.Layout.Column = 3;
            obj.sessionEdit.Value = 'C:\behavior';

            obj.videoGrid = uigridlayout(obj.grid,[1 2]);
            obj.videoGrid.Layout.Row = 2; obj.videoGrid.Layout.Column = [1 3];
            obj.videoGrid.RowHeight = {'1x'};
            obj.videoGrid.ColumnWidth = {'1x','1x'};
            obj.videoGrid.ColumnSpacing = 10;

            obj.sessionVideoPanel = uipanel(obj.videoGrid,'Title','Session Video');
            obj.sessionVideoPanel.Layout.Row = 1; obj.sessionVideoPanel.Layout.Column = 1;
            obj.sessionVideoPanelGrid = uigridlayout(obj.sessionVideoPanel,[2 1], ...
                'RowHeight',{'1x','fit'},'ColumnWidth',{'1x'});

            obj.sessionVideoAxes = uiaxes(obj.sessionVideoPanelGrid);
            axis(obj.sessionVideoAxes,'off'); title(obj.sessionVideoAxes,'Session Video');

            obj.selectROIButton = uibutton(obj.sessionVideoPanelGrid,'Text','Select ROI', ...
                'ButtonPushedFcn',@obj.selectROI);
            obj.selectROIButton.Layout.Row = 2;

            obj.binarizedVideoPanel = uipanel(obj.videoGrid,'Title','Binarized Video');
            obj.binarizedVideoPanel.Layout.Row = 1; obj.binarizedVideoPanel.Layout.Column = 2;
            obj.binarizedVideoPanelGrid = uigridlayout(obj.binarizedVideoPanel,[4 1], ...
                'RowHeight',{'1x','fit','fit','fit'},'ColumnWidth',{'1x'});

            obj.binarizedVideoAxes = uiaxes(obj.binarizedVideoPanelGrid);
            axis(obj.binarizedVideoAxes,'off'); title(obj.binarizedVideoAxes,'Binarized Video');

            obj.fecThreshLabel = uilabel(obj.binarizedVideoPanelGrid,'Text','FEC Binarization Threshold', ...
                'HorizontalAlignment','center');
            obj.fecThreshLabel.Layout.Row = 2;

            obj.fecThreshSlider = uislider(obj.binarizedVideoPanelGrid,...
                'Limits',[0 255],'Value',obj.binarizationThreshold, ...
                'ValueChangedFcn',@(src,~)obj.updateFECBinarizationThreshold(src.Value));
            obj.fecThreshSlider.Layout.Row = 3;

            obj.findEyeOpenMaxButton = uibutton(obj.binarizedVideoPanelGrid,'Text','Find Eye Open Max', ...
                'ButtonPushedFcn',@obj.findEyeOpenMax);
            obj.findEyeOpenMaxButton.Layout.Row = 4;



            obj.videoControlsGrid = uigridlayout(obj.grid,[1 3]);
            obj.videoControlsGrid.Layout.Row = 3;
            obj.videoControlsGrid.Layout.Column = [1 3];
            obj.videoControlsGrid.ColumnWidth = {'fit','fit','1x'};
            obj.videoControlsGrid.ColumnSpacing = 10;

            obj.playButton = uibutton(obj.videoControlsGrid,'Text','Play', ...
                'ButtonPushedFcn',@obj.playVideo);
            obj.playButton.Layout.Column = 1;

            obj.stopButton = uibutton(obj.videoControlsGrid,'Text','Stop', ...
                'ButtonPushedFcn',@obj.stopVideo);
            obj.stopButton.Layout.Column = 2;

            obj.videoSlider = uislider(obj.videoControlsGrid,'Limits',[0 1], ...
                'Value',0,'Enable','off', ...
                'ValueChangedFcn',@obj.videoSliderChanged, ...
                'ValueChangingFcn',@obj.videoSliderChanging);
            obj.videoSlider.Layout.Column = 3;

            obj.leftPanel = uipanel(obj.grid,'Title','Signal Options');
            obj.leftPanel.Layout.Row = 4; obj.leftPanel.Layout.Column = 1;
            obj.rightPanel = uipanel(obj.grid,'Title','Output Options');
            obj.rightPanel.Layout.Row = 4; obj.rightPanel.Layout.Column = 3;

            obj.buttonGrid = uigridlayout(obj.grid,[1 4]);
            obj.buttonGrid.Layout.Row = 5; obj.buttonGrid.Layout.Column = [1 3];
            obj.buttonGrid.ColumnWidth = {'1x','1x','1x','1x'};
            obj.buttonGrid.RowHeight = {'fit'};
            obj.buttonGrid.ColumnSpacing = 10;

            obj.loadDataButton = uibutton(obj.buttonGrid,'Text','Load Session Data', ...
                'ButtonPushedFcn',@obj.loadSessionData);
            obj.loadDataButton.Layout.Column = 1;

            obj.loadVideoButton = uibutton(obj.buttonGrid,'Text','Load Session Video', ...
                'ButtonPushedFcn',@obj.loadSessionVideo);
            obj.loadVideoButton.Layout.Column = 2;

            obj.processButton = uibutton(obj.buttonGrid,'Text','Process', ...
                'ButtonPushedFcn',@obj.runProcessing);
            obj.processButton.Layout.Column = 3;

            obj.exportButton = uibutton(obj.buttonGrid,'Text','Export Results');
            obj.exportButton.Layout.Column = 4;
        end

        function selectSessionFolder(obj,~,~)
            startPath = obj.sessionEdit.Value;
            if ~isfolder(startPath)
                startPath = pwd;
            end
            newPath = uigetdir(startPath,'Select Session Folder');
            if newPath ~= 0
                obj.sessionEdit.Value = newPath;
            end
            figure(obj.fig); drawnow;
        end

        function loadSessionData(obj,~,~)
            startPath = fullfile(obj.sessionEdit.Value,'session_data');
            if ~isfolder(startPath)
                startPath = obj.sessionEdit.Value;
            end
            if ~isfolder(startPath)
                startPath = pwd;
            end
            [file,path] = uigetfile({'*.mat;*.m','MATLAB Session Files';'*.*','All Files'},...
                'Load Session Data', startPath);
            if isequal(file,0)
                figure(obj.fig); drawnow;
                return;
            end
            figure(obj.fig); drawnow;
            fullPath = fullfile(path,file);
            [~,~,ext] = fileparts(fullPath);
            switch lower(ext)
                case '.mat'
                    warnState = warning('off','MATLAB:load:cannotInstantiateLoadedFunctionHandle');
                    cleanup = onCleanup(@() warning(warnState));
                    loaded = load(fullPath,'SessionData');
                    clear cleanup
                    if isfield(loaded,'SessionData')
                        obj.SessionData = loaded.SessionData;
                    else
                        obj.SessionData = loaded;
                    end
                otherwise
                    error('Unsupported session file type: %s', ext);
            end
            obj.sessionDataPath = fullPath;

            if isfield(obj.SessionData,'daqDataAll') && size(obj.SessionData.daqDataAll,1) >= 2
                obj.daqTrialSync = obj.SessionData.daqDataAll(1,:);
                obj.daqCamStrobe = obj.SessionData.daqDataAll(2,:);
            else
                obj.daqTrialSync = [];
                obj.daqCamStrobe = [];
            end
            if isfield(obj.SessionData,'daqTimestampsAll')
                obj.daqTimestampsAll = obj.SessionData.daqTimestampsAll(:).';
            else
                obj.daqTimestampsAll = [];
            end            
        end

        function sBin = binarizeSignal(~, sig, thresh)
            if nargin < 3 || isempty(thresh), thresh = 0.5; end
            if isempty(sig)
                sBin = [];
            else
                sBin = sig(:) > thresh;
            end
        end

        % function times = risingEdgeTimes(~, sig, t)
        %     if isempty(sig) || isempty(t)
        %         times = [];
        %         return;
        %     end
        %     s = logical(sig(:));          % assume already binarized
        %     idx = find(diff(s) == 1) + 1; % rising edges
        %     idx = idx(idx <= numel(t));
        %     times = t(idx);
        % end

        function dropped = findDroppedFrameNumbers(obj)
            fn = obj.frameNumArr;
            if isempty(fn)
                dropped = [];
                return;
            end
            expected = fn(1):fn(end);
            dropped = setdiff(expected, fn(:).');
            obj.droppedFrameNumbers = dropped(:);
        end

        function adjustCamStrobeTimes(obj)
            if isempty(obj.camStrobeRiseTimes)
                obj.camStrobeRiseTimesAdjusted = [];
                return;
            end
            if isempty(obj.droppedFrameNumbers)
                obj.camStrobeRiseTimesAdjusted = obj.camStrobeRiseTimes;
                return;
            end
            mask = true(numel(obj.camStrobeRiseTimes),1);
            dropIdx = obj.droppedFrameNumbers;
            dropIdx = dropIdx(dropIdx >= 1 & dropIdx <= numel(mask));
            mask(dropIdx) = false;
            obj.camStrobeRiseTimesAdjusted = obj.camStrobeRiseTimes(mask);
        end

        % function alignDaqToSession(obj)
        %     % Align DAQ timestamps to session clock using trial sync edges
        %     if isempty(obj.trialSyncRiseTimes) || ~isfield(obj.SessionData,'TrialStartTimestamp')
        %         obj.daqTimestampsAllSessionAligned = [];
        %         return;
        %     end
        %     syncDaq  = obj.trialSyncRiseTimes(:);
        %     syncSess = obj.SessionData.TrialStartTimestamp(:);

        %     n = min(numel(syncDaq), numel(syncSess));
        %     if n == 0
        %         obj.daqTimestampsAllSessionAligned = [];
        %         return;
        %     end
        %     syncDaq  = syncDaq(1:n);
        %     syncSess = syncSess(1:n);

        %     if n == 1
        %         offset = syncSess(1) - syncDaq(1);
        %         obj.daqTimestampsAllSessionAligned = obj.daqTimestampsAll + offset;
        %     else
        %         % piecewise linear warp
        %         obj.daqTimestampsAllSessionAligned = interp1(syncDaq, syncSess, obj.daqTimestampsAll, 'linear', 'extrap');
        %     end
        % end        

        function idx = risingEdgeIndices(~, sig)
            if isempty(sig)
                idx = [];
                return;
            end
            s = logical(sig(:));
            idx = find(diff(s) == 1) + 1;
        end

        function alignDaqToSession(obj, trialSyncIdx)
            % Align DAQ timestamps to session clock using trial sync edges
            if nargin < 2, trialSyncIdx = []; end
            if isempty(trialSyncIdx) || isempty(obj.daqTimestampsAll) || ~isfield(obj.SessionData,'TrialStartTimestamp')
                obj.daqTimestampsAllSessionAligned = [];
                return;
            end
            trialSyncIdx = trialSyncIdx(trialSyncIdx >= 1 & trialSyncIdx <= numel(obj.daqTimestampsAll));
            syncDaq = obj.daqTimestampsAll(trialSyncIdx);
            syncSess = obj.SessionData.TrialStartTimestamp(:);

            n = min(numel(syncDaq), numel(syncSess));
            if n == 0
                obj.daqTimestampsAllSessionAligned = [];
                return;
            end
            syncDaq  = syncDaq(1:n);
            syncSess = syncSess(1:n);

            if n == 1
                offset = syncSess(1) - syncDaq(1);
                obj.daqTimestampsAllSessionAligned = obj.daqTimestampsAll + offset;
            else
                obj.daqTimestampsAllSessionAligned = interp1(syncDaq, syncSess, obj.daqTimestampsAll, 'linear', 'extrap');
            end
        end

        function loadSessionVideo(obj,~,~)
            startPath = fullfile(obj.sessionEdit.Value,'video_data');
            if ~isfolder(startPath)
                startPath = obj.sessionEdit.Value;
            end
            if ~isfolder(startPath)
                startPath = pwd;
            end
            [file,path] = uigetfile({'*.mp4;*.avi;*.mov','Video Files';'*.*','All Files'},...
                'Load Session Video', startPath);
            if isequal(file,0)
                figure(obj.fig); drawnow;
                return;
            end
            figure(obj.fig); drawnow;
            fullPath = fullfile(path,file);
            try
                reader = VideoReader(fullPath);
                reader.CurrentTime = 0;
                frame = readFrame(reader);
                reader.CurrentTime = 0;

                obj.currentFrame = frame;
                obj.sessionVideoPath = fullPath;
                obj.sessionVideoReader = reader;
                obj.videoDuration = reader.Duration;
                obj.videoFrameRate = reader.FrameRate;
                obj.videoSlider.Value = 0;
                obj.videoSlider.Enable = 'on';

                obj.updateSessionVideo();
                obj.updateBinarizedVideo();
            catch ME
                obj.sessionVideoPath = '';
                obj.sessionVideoReader = [];
                obj.videoDuration = 0;
                obj.videoFrameRate = 0;
                obj.videoSlider.Value = 0;
                obj.videoSlider.Enable = 'off';
                cla(obj.sessionVideoAxes);
                uialert(obj.fig,sprintf('Failed to load video:\n%s',ME.message),'Video Error');
            end
        end

        function selectROI(obj,~,~)
            if isempty(obj.currentFrame)
                uialert(obj.fig,'Load a session video before selecting an ROI.','ROI Error');
                return;
            end
            delete(findall(obj.sessionVideoAxes,'Type','images.roi.Freehand'));
            if ~isempty(obj.roiHandle) && isvalid(obj.roiHandle)
                delete(obj.roiHandle);
                obj.roiHandle = [];
            end
            figure(obj.fig); drawnow;
            obj.roiHandle = drawfreehand(obj.sessionVideoAxes,'Color',[0.85 0.33 0.1],...
                'LineWidth',1.5,'Closed',true);
            % addlistener(obj.roiHandle,'MovingROI',@(~,~)obj.handleROIChanged());
            % addlistener(obj.roiHandle,'ROIMoved',@(~,~)obj.handleROIChanged());
            obj.handleROIChanged();
        end

        function handleROIChanged(obj)
            if isempty(obj.roiHandle) || ~isvalid(obj.roiHandle)
                obj.roiPosition = [];
                obj.roiMask = [];
                obj.updateBinarizedVideo();
                return;
            end
            obj.roiPosition = obj.roiHandle.Position;
            if isempty(obj.sessionVideoImage) || ~isgraphics(obj.sessionVideoImage,'image')
                obj.roiMask = createMask(obj.roiHandle);
            else
                obj.roiMask = createMask(obj.roiHandle,obj.sessionVideoImage);
            end
            obj.updateBinarizedVideo();
        end

        function updateFECBinarizationThreshold(obj,newValue)
            obj.binarizationThreshold = newValue;
            obj.updateBinarizedVideo();
        end

        function updateSessionVideo(obj)
            if isempty(obj.currentFrame)
                return;
            end
            obj.applyVideoDisplayScale();

            rgbFrame = obj.currentFrame;
            if size(rgbFrame,3) == 1
                rgbFrame = repmat(rgbFrame,1,1,3);
            end
            % rgbFrame = im2double(rgbFrame);

            if isempty(obj.sessionVideoImage) || ~isvalid(obj.sessionVideoImage)
                obj.sessionVideoImage = image(obj.sessionVideoAxes,rgbFrame);
                axis(obj.sessionVideoAxes,'image','off');
            else
                set(obj.sessionVideoImage,'CData',rgbFrame);
            end
        end

        function binFrame = binarizeImage(obj)
            binFrame = [];
            if isempty(obj.currentFrame) || isempty(obj.roiMask)
                return;
            end
            grayFrame = im2gray(obj.currentFrame);
            mask = obj.roiMask;
            % if ~isequal(size(mask),size(grayFrame))
            %     mask = imresize(mask,size(grayFrame));
            % end
            grayFrame(~mask) = 0;
            binFrame = imbinarize(grayFrame, obj.binarizationThreshold/255);
            obj.binFrame = binFrame;
        end

        function updateBinarizedVideo(obj)
            binFrame = obj.binarizeImage();
            if isempty(binFrame)
                cla(obj.binarizedVideoAxes);
                axis(obj.binarizedVideoAxes,'off');
                obj.binarizedVideoImage = [];
                return;
            end
            obj.applyVideoDisplayScale();
            % binFrame = im2double(binFrame);

            if isempty(obj.binarizedVideoImage) || ~isvalid(obj.binarizedVideoImage)
                obj.binarizedVideoImage = image(obj.binarizedVideoAxes, binFrame);
                colormap(obj.binarizedVideoAxes,gray(2));
                axis(obj.binarizedVideoAxes,'image','off');
            else
                set(obj.binarizedVideoImage,'CData',binFrame);
            end
        end

        function applyVideoDisplayScale(obj)
            if isempty(obj.currentFrame)
                return;
            end
            [frameH,frameW,~] = size(obj.currentFrame);
            scaledH = max(50, round(frameH * obj.videoDisplayScale));
            scaledW = max(50, round(frameW * obj.videoDisplayScale));

            obj.resizeAxes(obj.sessionVideoAxes, scaledW, scaledH);
            obj.resizeAxes(obj.binarizedVideoAxes, scaledW, scaledH);
        end

        function resizeAxes(~, ax, width, height)
            oldUnits = ax.Units;
            ax.Units = 'pixels';
            pos = ax.Position;
            pos(3) = width;
            pos(4) = height;
            ax.Position = pos;
            ax.Units = oldUnits;
        end


        function playVideo(obj,~,~)
            if isempty(obj.sessionVideoReader)
                return;
            end
            obj.stopVideo();
            % period = 1/max(1,obj.videoFrameRate);
            period = 1/max(1,30);
            obj.isPlaying = true;
            obj.playbackTimer = timer('ExecutionMode','fixedRate','Period',period,...
                'TimerFcn',@(~,~)obj.advanceFrame,'ErrorFcn',@(~,~)obj.stopVideo);
            start(obj.playbackTimer);
        end

        function stopVideo(obj,~,~)
            obj.isPlaying = false;
            if ~isempty(obj.playbackTimer) && isvalid(obj.playbackTimer)
                stop(obj.playbackTimer);
                delete(obj.playbackTimer);
            end
            obj.playbackTimer = [];
        end

        function advanceFrame(obj)
            if isempty(obj.sessionVideoReader)
                obj.stopVideo();
                return;
            end
            if hasFrame(obj.sessionVideoReader)
                frame = readFrame(obj.sessionVideoReader);
                obj.currentFrame = frame;
                obj.updateSessionVideo();
                obj.updateBinarizedVideo();
                obj.updateSliderFromReader();
            else
                obj.stopVideo();
                if obj.videoDuration > 0
                    obj.sessionVideoReader.CurrentTime = 0;
                    obj.videoSlider.Value = 0;
                end
            end
        end

        function videoSliderChanging(obj,~,evt)
            obj.seekVideo(evt.Value);
        end

        function videoSliderChanged(obj,src,~)
            obj.seekVideo(src.Value);
        end

        function seekVideo(obj,value)
            if isempty(obj.sessionVideoReader) || obj.videoDuration <= 0
                return;
            end
            clamped = min(max(value,0),1);
            wasPlaying = obj.isPlaying;
            obj.stopVideo();

            targetTime = clamped * obj.videoDuration;
            if targetTime >= obj.videoDuration
                targetTime = max(0,obj.videoDuration - 1/max(1,obj.videoFrameRate));
            end
            obj.sessionVideoReader.CurrentTime = targetTime;

            if hasFrame(obj.sessionVideoReader)
                frame = readFrame(obj.sessionVideoReader);
                obj.currentFrame = frame;
                obj.updateSessionVideo();
                obj.updateBinarizedVideo();
                obj.videoSlider.Value = clamped;
            end

            if wasPlaying
                obj.playVideo();
            end
        end

        function updateSliderFromReader(obj)
            if isempty(obj.sessionVideoReader) || obj.videoDuration <= 0
                return;
            end
            pos = min(max(obj.sessionVideoReader.CurrentTime / obj.videoDuration,0),1);
            obj.videoSlider.Value = pos;
        end

        function delete(obj)
            obj.stopVideo();
        end

        function fec = calculateRawFEC(obj, frame)
            if isempty(obj.roiMask)
                error('ROI mask is empty. Select ROI first.');
            end
            if nargin < 2 || isempty(frame)
                error('Frame is empty.');
            end
            mask = obj.roiMask;
            % if size(mask,1) ~= size(frame,1) || size(mask,2) ~= size(frame,2)
            %     mask = imresize(mask, [size(frame,1) size(frame,2)]);
            % end
            grayFrame = im2gray(frame);
            grayFrame(~mask) = 0;
            obj.binFrame = imbinarize(grayFrame, obj.binarizationThreshold/255);

            obj.totalEllipsePixels = nnz(mask == 1);
            obj.eyeAreaPixels = sum(obj.binFrame(mask == 1) == 0);

            fec = 1 - (obj.eyeAreaPixels / obj.totalEllipsePixels);
        end

        function fec = calculateAdjustedFEC(obj, eyeAreaPixels, totalEllipsePixels)
            if isempty(obj.minFur)
                error('minFur is empty. Run findEyeOpenMax after raw FEC.');
            end
            denom = totalEllipsePixels - obj.minFur;
            if denom <= 0
                error('Invalid denominator (totalEllipsePixels - minFur <= 0).');
            end
            fec = 1 - (eyeAreaPixels / denom);
        end

        function findEyeOpenMax(obj,varargin)
            if isempty(obj.fecDataRaw)
                error('No raw FEC data available. Run processing first.');
            end
            obj.minFur = min(obj.ellipsePixelSeries - obj.eyeAreaSeries);
        end

        function [A,B,C,D] = extractTimestamp(obj, frame)
            if isempty(frame)
                error('Frame is empty.');
            end
            % Take first four bytes (big-endian) from the raw frame buffer

            x = frame(1, 1:4);

            word = uint32(0);
            word = bitor(word, bitshift(uint32(x(1)), 24));
            word = bitor(word, bitshift(uint32(x(2)), 16));
            word = bitor(word, bitshift(uint32(x(3)),  8));
            word = bitor(word, uint32(x(4)));

            A = bitshift(word, -25);                    % top 7 bits (seconds)
            B = bitand(bitshift(word, -12), 2^13 - 1);  % next 13 bits (cycle count)
            C = bitand(word, (2^12 - 1) - (2^4-1));     % lower 12 bits, low 4 zeroed (cycle offset)
            D = bitand(word, 2^4 - 1);                  % lower 4 bits (frame number nibble)
        end

        function runProcessing(obj, varargin)
            if isempty(obj.sessionVideoReader)
                uialert(obj.fig,'Load a session video first.','Process Error');
                return;
            end
            if isempty(obj.roiMask)
                uialert(obj.fig,'Select an ROI before processing.','Process Error');
                return;
            end

            reader = obj.sessionVideoReader;
            reader.CurrentTime = 0;

            estFrames = max(1, floor(reader.Duration * reader.FrameRate));
            d = uiprogressdlg(obj.fig,'Title','Processing','Message','0% complete','Cancelable','on');

            fecRaw = [];
            ellipsePixels = [];
            eyeAreas = [];
            secArr = [];
            cycArr = [];
            offArr = [];
            nibArr = [];            
            frameIdx = 1;
            % initFrameIdx = frameIdx;
            % estFrames = 10000;
            try
                % process each image
                while hasFrame(reader)
                % while frameIdx <= estFrames
                    if d.CancelRequested
                        break;
                    end
                    frame = readFrame(reader);
                    fecRaw(frameIdx,1) = obj.calculateRawFEC(frame); %#ok<AGROW>
                    ellipsePixels(frameIdx,1) = obj.totalEllipsePixels; %#ok<AGROW>
                    eyeAreas(frameIdx,1) = obj.eyeAreaPixels; %#ok<AGROW>

                    [A,B,C,D] = obj.extractTimestamp(frame);
                    secArr(frameIdx,1) = A; %#ok<AGROW>
                    cycArr(frameIdx,1) = B; %#ok<AGROW>
                    offArr(frameIdx,1) = C; %#ok<AGROW>
                    nibArr(frameIdx,1) = D; %#ok<AGROW>


                    d.Value = min(frameIdx/estFrames,1);
                    d.Message = sprintf('%.0f%% complete', 100*d.Value);
                    frameIdx = frameIdx + 1;
                end
            catch ME
                close(d);
                rethrow(ME);
            end
            close(d);

            obj.fecDataRaw = fecRaw;
            obj.ellipsePixelSeries = ellipsePixels;
            obj.eyeAreaSeries = eyeAreas;
            obj.secondCount = secArr;
            obj.cycleCount = cycArr;
            obj.cycleOffset = offArr;
            obj.frameNibble = nibArr;

            unwrapNib = nibArr(:);
            wrapOffset = 0;
            for ii = 2:numel(unwrapNib)
                if unwrapNib(ii) < nibArr(ii-1)
                    wrapOffset = wrapOffset + 16;
                end
                unwrapNib(ii) = unwrapNib(ii) + wrapOffset;
            end
            unwrapNib = unwrapNib - (unwrapNib(1) - 1);
            obj.frameNumArr = unwrapNib;
            % obj.frameNumArr = unwrapNib + (initFrameIdx - 1);


            % compute timestamp (seconds + cycle/7999)
            cycleSeconds = double(obj.cycleCount) / 7999;
            obj.timestampSec = double(obj.secondCount) + cycleSeconds;
                    
            % binarize DAQ
            binTrial = obj.binarizeSignal(obj.daqTrialSync);
            binStrobe = obj.binarizeSignal(obj.daqCamStrobe);

            % rising-edge indices
            trialIdx  = obj.risingEdgeIndices(binTrial);
            strobeIdx = obj.risingEdgeIndices(binStrobe);

            % trim cam strobe if trailing pulses
            % cam strobe pulses restart after stop(vid) at session end as
            % cam switches to 'freerunning' without storing images
            % trailing pulses aren't actual images in video

            %  BpodSystem.Data.TriggerPulseCount 
            [strobeMaxDiff, strobeMaxDiffIdx] = max(diff(strobeIdx));
            if strobeMaxDiff > 30
                strobeIdx = strobeIdx(1:strobeMaxDiffIdx);
            end

            % align DAQ clock to session clock using trial sync edges
            obj.alignDaqToSession(trialIdx);

            % use aligned timestamps for edge times (fallback to raw if alignment failed)
            tsAligned = obj.daqTimestampsAllSessionAligned;
            if isempty(tsAligned)
                tsAligned = obj.daqTimestampsAll;
            end

            trialIdx  = trialIdx(trialIdx >= 1 & trialIdx <= numel(tsAligned));
            strobeIdx = strobeIdx(strobeIdx >= 1 & strobeIdx <= numel(tsAligned));

            obj.trialSyncRiseTimes = tsAligned(trialIdx);
            obj.camStrobeRiseTimes  = tsAligned(strobeIdx);

            % dropped frames and adjusted strobe times
            obj.findDroppedFrameNumbers();   %  obj.droppedFrameNumbers
            obj.adjustCamStrobeTimes();  % removes missing image rise times, obj.camStrobeRiseTimesAdjusted

            % exposure center per image
            obj.FECTimes = obj.camStrobeRiseTimesAdjusted + 0.0015;            

            obj.findEyeOpenMax();

            % Adjusted FEC using stored eye areas
            totalEllipse = obj.totalEllipsePixels;
            obj.FEC = arrayfun(@(ea)obj.calculateAdjustedFEC(ea,totalEllipse), eyeAreas);

            % paper testing shows ~20ms delay for air after opening valve.
            % need to quantify more accurately with statistical tests
            % across session
            valveDelay = 0.02;        % 20ms delay for air travel
            obj.SessionData.valveDelay = valveDelay;            

            % Backward-compatible per-trial segmentation
            nTrials = obj.SessionData.nTrials;
            nStarts = numel(obj.trialSyncRiseTimes);
            for trial = 1:nTrials
            % for trial = 1:6
                if trial > nStarts
                    obj.SessionData.RawEvents.Trial{1,trial}.Data.FECTimes = [];
                    obj.SessionData.RawEvents.Trial{1,trial}.Data.FEC = [];
                    continue;
                end
                tStart = obj.trialSyncRiseTimes(trial);
                tEnd   = obj.SessionData.TrialEndTimestamp(trial);
                idxWin = obj.FECTimes >= tStart & obj.FECTimes <= tEnd;
                obj.SessionData.RawEvents.Trial{1,trial}.Data.FECTimes = obj.FECTimes(idxWin) - tStart;
                obj.SessionData.RawEvents.Trial{1,trial}.Data.FEC      = obj.FEC(idxWin);
                obj.SessionData.RawEvents.Trial{1,trial}.Data.totalEllipsePixels = totalEllipse;
                obj.SessionData.RawEvents.Trial{1,trial}.Data.eyeAreaPixels = eyeAreas(idxWin);
                obj.SessionData.RawEvents.Trial{1,trial}.Data.minFur = obj.minFur;
                obj.SessionData.RawEvents.Trial{1,1}.Events.AirContact = obj.SessionData.RawEvents.Trial{1,1}.Events.GlobalTimer2_Start + valveDelay;               
            end



            figure;
            hold on;
            % plot(obj.FECTimes(1:2000), obj.FEC);
            % plot(obj.FECTimes(1:length(obj.FEC)), obj.FEC, 'b', 'DisplayName','FEC');
            % plot(obj.FECTimes, obj.FEC(1:length(obj.FECTimes)), 'b', 'DisplayName','FEC');
            plot(obj.FECTimes, obj.FEC, 'b', 'DisplayName','FEC');

            led_time = obj.SessionData.RawEvents.Trial{1,1}.Events.GlobalTimer1_Start + obj.SessionData.TrialStartTimestamp(1);
            ap_time = obj.SessionData.RawEvents.Trial{1,1}.Events.GlobalTimer2_Start + obj.SessionData.TrialStartTimestamp(1);
            ap_actual_time = ap_time + valveDelay;
            xl1 = xline(led_time, '--b', 'DisplayName', 'LED Turned On');
            xl2 = xline(ap_time, '--r', 'DisplayName', 'Air Puff');
            xl3 = xline(ap_actual_time, '--g', 'DisplayName', 'Air Puff Actual');

            legend show;
            hold off;

            figure();
            hold on;
            % p1 = plot(obj.FECTimes, obj.FEC, 'DisplayName','FEC');
            p1 = plot(obj.FECTimes(1:length(obj.FEC)), obj.FEC, 'b', 'DisplayName','FEC');

            for trial = (1:obj.SessionData.nTrials)
            % for trial = (1:4)
                led_time = obj.SessionData.RawEvents.Trial{1,trial}.Events.GlobalTimer1_Start + obj.SessionData.TrialStartTimestamp(trial);
                ap_time = obj.SessionData.RawEvents.Trial{1,trial}.Events.GlobalTimer2_Start + obj.SessionData.TrialStartTimestamp(trial);
                ap_actual_time = ap_time + valveDelay;
                start_time = obj.SessionData.RawEvents.Trial{1,trial}.States.Start(1) + obj.SessionData.TrialStartTimestamp(trial);
                iti_pre = obj.SessionData.RawEvents.Trial{1,trial}.States.ITI_Pre(1) + obj.SessionData.TrialStartTimestamp(trial);
                check_eye_open = obj.SessionData.RawEvents.Trial{1,trial}.States.CheckEyeOpen(1) + obj.SessionData.TrialStartTimestamp(trial);

                xl1 = xline(led_time, '--b', 'DisplayName', 'LED Turned On');
                xl2 = xline(ap_time, '--r', 'DisplayName', 'Air Puff');
                xl3 = xline(ap_actual_time, '--g', 'DisplayName', 'Air Puff Actual');
                xl4 = xline(start_time, '--b', 'DisplayName', 'Start');
                xl5 = xline(iti_pre, '--r', 'DisplayName', 'ITI_Pre');
                xl6 = xline(check_eye_open, '--g', 'DisplayName', 'Check_Eye_Open');
            end
            % legend([p1 xl1 xl2 xl3], {'FEC', 'LED Turned On', 'Air Puff', 'Air Puff Actual'});
            legend([p1 xl1 xl2 xl3 xl4 xl5 xl6], {'FEC', 'LED Turned On', 'Air Puff', 'Air Puff Actual', 'Start', 'ITI_Pre', 'Check_Eye_Open'});
            % legend show;
            hold off;            

            % for trial = (1:obj.SessionData.nTrials)
            for trial = (1:4)
                figure;
                time = obj.SessionData.RawEvents.Trial{1,trial}.Data.FECTimes;
                fec = obj.SessionData.RawEvents.Trial{1,trial}.Data.FEC;
                plot(time, fec, 'DisplayName','FEC');
                hold on;
                led_time = obj.SessionData.RawEvents.Trial{1,trial}.Events.GlobalTimer1_Start;
                ap_time = obj.SessionData.RawEvents.Trial{1,trial}.Events.GlobalTimer2_Start;
                ap_actual_time = ap_time + valveDelay;

                xl1 = xline(led_time, '--b', 'DisplayName', 'LED Turned On');
                xl2 = xline(ap_time, '--r', 'DisplayName', 'Air Puff');
                xl3 = xline(ap_actual_time, '--g', 'DisplayName', 'Air Puff Actual');

                legend show;
                hold off;    
            end   
        

            % % for trial = (1:ans.SessionData.nTrials)
            % for trial = (185:188)
            %     figure;
            %     time = ans.SessionData.RawEvents.Trial{1,trial}.Data.FECTimes;
            %     fec = ans.SessionData.RawEvents.Trial{1,trial}.Data.FEC;
            %     plot(time, fec, 'DisplayName','FEC');
            %     hold on;
            %     led_time = ans.SessionData.RawEvents.Trial{1,trial}.Events.GlobalTimer1_Start;
            %     ap_time = ans.SessionData.RawEvents.Trial{1,trial}.Events.GlobalTimer2_Start;
            %     ap_actual_time = ap_time + valveDelay;
            % 
            %     xl1 = xline(led_time, '--b', 'DisplayName', 'LED Turned On');
            %     xl2 = xline(ap_time, '--r', 'DisplayName', 'Air Puff');
            %     xl3 = xline(ap_actual_time, '--g', 'DisplayName', 'Air Puff Actual');
            % 
            %     legend show;
            %     hold off;    
            % end               

            % 
            SessionData = obj.SessionData;
            % save(obj.sessionDataPath, '-struct', 'S');
            save(obj.sessionDataPath, 'SessionData');
            % save(obj.sessionDataPath, 'SessionData', '-append');
            

            uialert(obj.fig,sprintf('FEC computed for %d frames.', numel(fecRaw)),'Processing Complete');
        end

     
    end
end