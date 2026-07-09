function app = EBC_PostProcess_V_5_3()
    app.fig = uifigure('Name','EBC Post Process','Position',[100 100 900 600]);
    app.SessionData = [];
    app.sessionDataPath = '';
    app.sessionVideoPath = '';
    app.sessionVideoReader = [];
    app.sessionVideoImage = [];
    app.binarizedVideoImage = [];
    app.currentFrame = [];
    app.binFrame = [];
    app.binarizationThreshold = 40;
    app.roiHandle = [];
    app.roiPosition = [];
    app.roiMask = [];

    % [warnMsg, warnID] = inspectwarn   % use to find warnings that popup
    warning('off','MATLAB:dispatcher:UnresolvedFunctionHandle');


    
    % 3 rows x 2 columns grid with shared padding
    % app.grid = uigridlayout(app.fig,[3 2]);
    % app.grid.RowHeight = {'fit','1x','fit'};
    % app.grid.ColumnWidth = {'1x','1x'};
    app.grid = uigridlayout(app.fig,[4 3]);
    app.grid.RowHeight = {'fit','2x','1x','fit'};
    app.grid.ColumnWidth = {'fit','fit','1x'};
    app.grid.Padding = [15 15 15 15];
    app.grid.RowSpacing = 10;
    app.grid.ColumnSpacing = 10;

    % Row 1 controls
    % app.sessionLabel = uilabel(app.grid,'Text','Session Folder:','HorizontalAlignment','right');
    % app.sessionLabel.Layout.Row = 1; app.sessionLabel.Layout.Column = 1;

    % app.sessionEdit = uieditfield(app.grid,'text');
    % app.sessionEdit.Layout.Row = 1; app.sessionEdit.Layout.Column = 2;
    % app.sessionEdit = uieditfield(app.grid,'text');
    % app.sessionEdit.Layout.Row = 1; app.sessionEdit.Layout.Column = 2;
    % app.sessionEdit.Value = 'C:\behavior\session_data';
    app.sessionLabel = uilabel(app.grid,'Text','Session Folder:','HorizontalAlignment','right');
    app.sessionLabel.Layout.Row = 1; app.sessionLabel.Layout.Column = 1;

    app.sessionEdit = uieditfield(app.grid,'text');
    app.sessionEdit.Layout.Row = 1; app.sessionEdit.Layout.Column = 3;
    app.sessionEdit.Value = 'C:\behavior';    

    app.sessionBrowse = uibutton(app.grid,'push','Text','Browse...','ButtonPushedFcn',@(~,~)selectSessionFolder(app));
    app.sessionBrowse.Layout.Row = 1; app.sessionBrowse.Layout.Column = 2;

    % Row 2 video viewers (span full width)
    app.videoGrid = uigridlayout(app.grid,[1 2]);
    app.videoGrid.Layout.Row = 2;
    app.videoGrid.Layout.Column = [1 3];
    app.videoGrid.RowHeight = {'1x'};
    app.videoGrid.ColumnWidth = {'1x','1x'};
    app.videoGrid.ColumnSpacing = 10;

    app.sessionVideoPanel = uipanel(app.videoGrid,'Title','Session Video');
    app.sessionVideoPanel.Layout.Row = 1; app.sessionVideoPanel.Layout.Column = 1;
    app.sessionVideoPanelGrid = uigridlayout(app.sessionVideoPanel,[2 1], ...
        'RowHeight',{'1x','fit'},'ColumnWidth',{'1x'});
    app.sessionVideoAxes = uiaxes(app.sessionVideoPanelGrid);
    axis(app.sessionVideoAxes,'off'); title(app.sessionVideoAxes,'Session Video');
    app.selectROIButton = uibutton(app.sessionVideoPanelGrid,'Text','Select ROI', ...
        'ButtonPushedFcn',@(~,~)selectROI(app));
    app.selectROIButton.Layout.Row = 2;

    app.binarizedVideoPanel = uipanel(app.videoGrid,'Title','Binarized Video');
    app.binarizedVideoPanel.Layout.Row = 1; app.binarizedVideoPanel.Layout.Column = 2;
    app.binarizedVideoPanelGrid = uigridlayout(app.binarizedVideoPanel,[4 1], ...
        'RowHeight',{'1x','fit','fit','fit'},'ColumnWidth',{'1x'});
    app.binarizedVideoAxes = uiaxes(app.binarizedVideoPanelGrid);
    axis(app.binarizedVideoAxes,'off'); title(app.binarizedVideoAxes,'Binarized Video');

    app.fecThreshLabel = uilabel(app.binarizedVideoPanelGrid,'Text','FEC Binarization Threshold');
    app.fecThreshLabel.HorizontalAlignment = 'center'; app.fecThreshLabel.Layout.Row = 2;
    app.fecThreshSlider = uislider(app.binarizedVideoPanelGrid, ...
        'Limits',[0 255],'Value',app.binarizationThreshold, ...
        'ValueChangedFcn',@(src,~)updateFECBinarizationThreshold(app,src.Value));
    app.fecThreshSlider.Layout.Row = 3; app.fecThreshSlider.Layout.Column = 1;

    app.findEyeOpenMaxButton = uibutton(app.binarizedVideoPanelGrid,'Text','Find Eye Open Max', ...
        'ButtonPushedFcn',@(~,~)findEyeOpenMax(app));
    app.findEyeOpenMaxButton.Layout.Row = 4;

    % Row 2 panels
    app.leftPanel = uipanel(app.grid,'Title','Signal Options');
    app.leftPanel.Layout.Row = 3; app.leftPanel.Layout.Column = 1;

    app.rightPanel = uipanel(app.grid,'Title','Output Options');
    app.rightPanel.Layout.Row = 3; app.rightPanel.Layout.Column = 3;

    % Row 3 buttons (span both columns)
    app.buttonGrid = uigridlayout(app.grid,[1 4]);
    app.buttonGrid.Layout.Row = 4; app.buttonGrid.Layout.Column = [1 3];
    app.buttonGrid.ColumnWidth = {'1x','1x','1x','1x'};
    app.buttonGrid.RowHeight = {'fit'};
    app.buttonGrid.ColumnSpacing = 10;

    app.loadDataButton = uibutton(app.buttonGrid,'Text','Load Session Data',...
        'ButtonPushedFcn',@(~,~)loadSessionData(app));
    app.loadDataButton.Layout.Column = 1;

    app.loadVideoButton = uibutton(app.buttonGrid,'Text','Load Session Video',...
        'ButtonPushedFcn',@(~,~)loadSessionVideo(app));
    app.loadVideoButton.Layout.Column = 2;

    app.processButton = uibutton(app.buttonGrid,'Text','Process','ButtonPushedFcn',@runProcessing);
    app.processButton.Layout.Column = 3;

    app.exportButton = uibutton(app.buttonGrid,'Text','Export Results');
    app.exportButton.Layout.Column = 4;
end


function selectSessionFolder(app)
    startPath = app.sessionEdit.Value;
    if ~isfolder(startPath)
        startPath = pwd;
    end
    newPath = uigetdir(startPath,'Select Session Folder');
    if newPath ~= 0
        app.sessionEdit.Value = newPath;
    end
end


function loadSessionData(app)
    startPath = app.sessionEdit.Value;
    startPath = fullfile(startPath, 'session_data');
    if ~isfolder(startPath)
        startPath = pwd;
    end
    [file, path] = uigetfile({'*.mat;*.m','MATLAB Session Files';'*.*','All Files'},...
        'Load Session Data', startPath);
    if isequal(file,0)
        figure(app.fig); drawnow;
        return;
    end
    figure(app.fig); drawnow;
    fullPath = fullfile(path,file);
    [~,~,ext] = fileparts(fullPath);

    switch lower(ext)
        case '.mat'
            loaded = load(fullPath,'SessionData');
            if isfield(loaded,'SessionData')
                app.SessionData = loaded.SessionData;
            else
                app.SessionData = loaded;
            end
        % case '.m'
        %     prevDir = pwd;
        %     cleanup = onCleanup(@() cd(prevDir));
        %     cd(path);
        %     clear SessionData
        %     run(file);
        %     if ~exist('SessionData','var')
        %         error('Selected script did not define SessionData.');
        %     end
        %     app.SessionData = SessionData;
        otherwise
            error('Unsupported session file type: %s', ext);
    end

    app.sessionDataPath = fullPath;
end

function loadSessionVideo(app)
    startPath = app.sessionEdit.Value;
    startPath = fullfile(startPath, 'video_data');
    if ~isfolder(startPath)
        startPath = pwd;
    end
    [file, path] = uigetfile({'*.mp4;*.avi;*.mov','Video Files';'*.*','All Files'},...
        'Load Session Video', startPath);
    if isequal(file,0)
        figure(app.fig); drawnow;
        return;
    end
    figure(app.fig); drawnow;
    fullPath = fullfile(path,file);
    try
        reader = VideoReader(fullPath);
        reader.CurrentTime = 0;
        frame = readFrame(reader);
        reader.CurrentTime = 0;

        updateSessionVideo(app, frame);
        updateBinarizedVideo(app);

        app.sessionVideoPath = fullPath;
        app.sessionVideoReader = reader;
    catch ME
        app.sessionVideoPath = '';
        app.sessionVideoReader = [];
        cla(app.sessionVideoAxes);
        uialert(app.fig,sprintf('Failed to load video:\n%s',ME.message),'Video Error');
    end
end

function selectROI(app)
    if isempty(app.currentFrame)
        uialert(app.fig,'Load a session video before selecting an ROI.','ROI Error');
        return;
    end

    delete(findall(app.sessionVideoAxes,'Type','images.roi.Freehand'));

    if ~isempty(app.roiHandle) && isvalid(app.roiHandle)
        delete(app.roiHandle);
        app.roiHandle = [];
    end

    figure(app.fig); drawnow;

    app.roiHandle = drawfreehand(app.sessionVideoAxes,'Color',[0.85 0.33 0.1],...
        'LineWidth',1.5,'Closed',true);

    addlistener(app.roiHandle,'MovingROI',@(~,~)handleROIChanged(app));
    addlistener(app.roiHandle,'ROIMoved',@(~,~)handleROIChanged(app));

    handleROIChanged(app);
end

function handleROIChanged(app)
    if isempty(app.roiHandle) || ~isvalid(app.roiHandle)
        app.roiPosition = [];
        app.roiMask = [];
        updateBinarizedVideo(app);
        return;
    end
    app.roiPosition = app.roiHandle.Position;

    targetImage = [];
    if ~isempty(app.sessionVideoImage) && isgraphics(app.sessionVideoImage,'image')
        targetImage = app.sessionVideoImage;
    end
    if isempty(targetImage)
        app.roiMask = createMask(app.roiHandle);
    else
        app.roiMask = createMask(app.roiHandle,targetImage);
    end
    updateBinarizedVideo(app);
end

function updateFECBinarizationThreshold(app,newValue)
    app.binarizationThreshold = newValue;
    updateBinarizedVideo(app);
end

function updateSessionVideo(app,frame)
    if nargin >= 2 && ~isempty(frame)
        app.currentFrame = frame;
    end
    if isempty(app.currentFrame)
        return;
    end
    if isempty(app.sessionVideoImage) || ~isvalid(app.sessionVideoImage)
        app.sessionVideoImage = image(app.sessionVideoAxes,app.currentFrame);
        axis(app.sessionVideoAxes,'off');
    else
        set(app.sessionVideoImage,'CData',app.currentFrame);
    end
end

function binFrame = binarizeImage(app)
    binFrame = [];
    if isempty(app.currentFrame) || isempty(app.roiMask)
        return;
    end
    grayFrame = im2gray(app.currentFrame);
    mask = app.roiMask;
    if ~isequal(size(mask),size(grayFrame))
        mask = imresize(mask,size(grayFrame));
    end
    grayFrame(~mask) = 0;
    binFrame = imbinarize(grayFrame, app.binarizationThreshold/255);
    app.binFrame = binFrame;
end

function updateBinarizedVideo(app)
    binFrame = binarizeImage(app);
    if isempty(binFrame)
        cla(app.binarizedVideoAxes);
        axis(app.binarizedVideoAxes,'off');
        app.binarizedVideoImage = [];
        return;
    end
    if isempty(app.binarizedVideoImage) || ~isvalid(app.binarizedVideoImage)
        app.binarizedVideoImage = image(app.binarizedVideoAxes, binFrame);
        colormap(app.binarizedVideoAxes,gray(2));
        axis(app.binarizedVideoAxes,'off');
    else
        set(app.binarizedVideoImage,'CData',binFrame);
    end
end

function findEyeOpenMax(app)
    % TODO: implement eye-open-max routine
end