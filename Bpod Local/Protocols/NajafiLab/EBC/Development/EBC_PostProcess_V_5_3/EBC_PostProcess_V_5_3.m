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
            if ~isequal(size(mask),size(grayFrame))
                mask = imresize(mask,size(grayFrame));
            end
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

        function findEyeOpenMax(obj,varargin) %#ok<INUSD>
            % TODO: implement eye-open-max routine
        end

        function runProcessing(obj,varargin) %#ok<INUSD>
            % TODO: implement processing routine
        end
    end
end