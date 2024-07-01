
global MEV;
MEV = [];

MEV = EyelidAnalyzer;

% https://www.mathworks.com/help/parallel-computing/perform-data-acquisition-in-parallel-with-postprocessing.html
% parallel pool with 1 worker
parpool('Processes',1);
% To send information back from the worker to the MATLAB client, create a DataQueue object.
DataQueue = parallel.pool.DataQueue;
%  To display images every time they arrive from the DataQueue object, use afterEach.
afterEach(DataQueue, @updateVideoPara);
freq = 5;
fpara = parfeval(@getFrameFromCamera,0,DataQueue,freq);

MEV.startGUIVideo();




% if isempty(obj.vidTimer) || ~isvalid(obj.vidTimer)
%     obj.vidTimer = timer('TimerFcn','', 'ExecutionMode', 'fixedRate', 'Period', 0.065);
%     set(obj.vidTimer, 'TimerFcn', @(x,y)obj.getFrameFromCamera());
%     start(obj.vidTimer);
%     disp(['Video Update Timer Started']);
% end  

% get para pool and cancel para workers
% poolobj = gcp('nocreate');
% delete(poolobj);


% MEV.stopGUIVideo;
% 
% 
% MEV.connectVideoTrial;
% pause(0.1);
% MEV.startVideoTrial;
% 
% 
% %MEV.vid
% MEV.vid.FramesAcquired
% MEV.vid.FramesAvailable
% 
% isrunning(MEV.vid)


% Display all valid trigger configurations.
%triggerinfo(MEV.vid)
%MEV.src.TriggerMode
function updateVideoPara(img)     
    if isobject(MEV.roiHandle) && isvalid(MEV.roiHandle)  
        MEV.roiPosition = getPosition(MEV.roiHandle);  % Get ROI position before playing  
    end

    imshow(img, 'Parent', MEV.axOriginal);

    if isobject(MEV.roiHandle) && isvalid(MEV.roiHandle)
        setPosition(MEV.roiHandle, MEV.roiPosition);  % Reset the position for consistency
    elseif ~isempty(MEV.roiPosition)
        MEV.roiHandle = imellipse(MEV.axOriginal, MEV.roiPosition);  % Recreate ROI if not valid
    end

    MEV.updateBinaryVideo;
end


function getFrameFromCamera(DataQueue, freq)
    while true
        if isrunning(MEV.vid)
            if MEV.vid.FramesAvailable > 0
                MEV.vid.FramesAvailable
                [data, time, metadata] = getdata(MEV.vid);
                MEV.frame = data(:,:,:,end);
                frame = MEV.frame;
                if MEV.isZoom
                    if ~isempty(MEV.frame)
                        img_crop = MEV.frame(409:550, 575:730);
                        MEV.frame = img_crop;
                        % imshow(MEV.frame, 'Parent', MEV.axOriginal);
                    end
                else
                    % imshow(MEV.frame, 'Parent', MEV.axOriginal);
                end
                send(DataQueue, frame);
                flushdata(MEV.vid);
            if isobject(MEV.roiHandle) && isvalid(MEV.roiHandle)
                setPosition(MEV.roiHandle, MEV.roiPosition);  % Reset the position for consistency
            elseif ~isempty(MEV.roiPosition)
                MEV.roiHandle = imellipse(MEV.axOriginal, MEV.roiPosition);  % Recreate ROI if not valid
            end
    
            % MEV.updateBinaryVideo;
            % stop(MEV.vidTimer);
            % start(MEV.vidTimer);                    
            end
        else
            % disp(['Video Input Buffer Empty']);
        end
        pause(1/freq);
    end
end