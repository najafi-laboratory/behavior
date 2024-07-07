function EBC_V_2_9
try
    global BpodSystem
    global S
    global MEV
        
    %% Import scriptsBpod
    
    % m_Plotter = Plotter;
    m_InitGUI = InitGUI;
    % m_TrialConfig = TrialConfig;
    % m_Opto = OptoConfig(EnableOpto);
    
    %% Turn off Bpod LEDs
    
    % This code will disable the state machine status LED
    BpodSystem.setStatusLED(0);

    % get matlab version
    v_info = version;
    BpodSystem.Data.MatVer = version;
    
  
    %% Define parameters
    [S] = m_InitGUI.SetParams(BpodSystem);
    
  
    %% Initialize plots
    
    BpodParameterGUI('init', S); % Initialize parameter GUI plugin
    BpodSystem.ProtocolFigures.ParameterGUI.Position = [9 53 442 185];
     
 
    set(BpodSystem.ProtocolFigures.ParameterGUI, 'Position', [9 53 697 539]);
    
    %% state timing plot
    useStateTiming = true;  % Initialize state timing plot
    if ~verLessThan('matlab','9.5') % StateTiming plot requires MATLAB r2018b or newer
        useStateTiming = true;
        StateTiming();
    end
    
    MEV = EyelidAnalyzer(BpodSystem.GUIData.SubjectName);
    MEV.initPlotLines();

    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_EBC';

    % wait for parameter update and confirm before beginning trial loop
    input('Set parameters and press enter to continue >', 's'); 
    S = BpodParameterGUI('sync', S);
       
    %% init any needed experimenter trial info values
    ExperimenterTrialInfo.TrialNumber = 1;

    %% Main trial loop    

    MaxTrials = 1000;

    MEV.stopPreTrialVideo();
    
    for currentTrial = 1:MaxTrials        
        %% sync trial-specific parameters from GUI
      
        % input('Set parameters and press enter to continue >', 's');
        S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
        S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin   

        %% Eye Analyzer        
        % move these to trial start function

        MEV.setTrialData();
        MEV.setEventTimes(S.GUI.LED_OnsetDelay, S.GUI.AirPuff_OnsetDelay, S.GUI.ITI_Pre);
        MEV.startTrialsVideo(currentTrial, BpodSystem.GUIData.SubjectName);
        MEV.plotLEDOnset();
        MEV.plotAirPuffOnset();


        %% experimenter info for console display

        ExperimenterTrialInfo.TrialNumber = currentTrial;   % check variable states as field/value struct for experimenter info
        strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
        disp(strExperimenterTrialInfo);          
    
        %% construct state matrix
    
        sma = NewStateMatrix(); % Assemble state matrix
                       
        % StimDur = max(S.GUI.LED_Dur, S.GUI.AirPuff_OnsetDelay + S.GUI.AirPuff_Dur);

        LED_offset = S.GUI.LED_OnsetDelay + S.GUI.LED_Dur;
        ISI = S.GUI.AirPuff_OnsetDelay - LED_offset;
                  
        if ISI <=0 % classical conditioning: no gap between led offset and puff onset
	        LED_puff = S.GUI.LED_Dur;

        else % trace conditioning: puff onset happens after led offset
	        LED_puff = S.GUI.LED_Dur + ISI + S.GUI.AirPuff_Dur;
        end

         vidDur = S.GUI.ITI_Pre + S.GUI.LED_OnsetDelay + LED_puff + S.GUI.ITI_Post; % unless LED dur is later than puff offset

        % LED Timer
        % LED Timer generated using behavior port 1 Pulse Width Modulation 
        % pin (PWM).  Starts after S.GUI.LED_OnsetDelay, on during
        % S.GUI.LED_Dur.
        sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', S.GUI.LED_Dur, 'OnsetDelay', S.GUI.LED_OnsetDelay,...
            'Channel', 'PWM1', 'PulseWidthByte', 255, 'PulseOffByte', 0,...
            'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0); 

        
        % Air Puff Timer
        % Air Puff Timer uses behavior port 1 12V valve pin.
        sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', S.GUI.AirPuff_Dur, 'OnsetDelay', S.GUI.AirPuff_OnsetDelay, 'Channel', 'Valve1',...
                     'OnMessage', 1, 'OffMessage', 0); 
               

        % t = S.GUI.ITI_Pre + S.GUI.LED_OnsetDelay + S.GUI.LED_Dur + 
        % fCam = 400;
        % T = 1 / fCam;
        % Dur = T / 2;
        % Set timer pulses for 250 fps video trigger
        CamTrigOnDur = 0.0005;
        CamTrigOffDur = 0.004 - CamTrigOnDur; %0.0025 - CamTrigOnDur;

        % Cam Trigger
        % Camera Trigger generated using BNC1 output.  400Hz signal.
        sma = SetGlobalTimer(sma, 'TimerID', 3, 'Duration', CamTrigOnDur, 'OnsetDelay', 0, 'Channel', 'BNC1',...
            'OnLevel', 1, 'OffLevel', 0,...
            'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', CamTrigOffDur);         

        % sma = AddState(sma, 'Name', 'CheckEyeOpen', ...
        %     'Timer', 0,...
        %     'StateChangeConditions', {'SoftCode1', 'Trigger', 'SoftCode2', 'Start'},...
        %     'OutputActions', {'SoftCode', 9});

        sma = AddState(sma, 'Name', 'Start', ...
            'Timer', 0,...
            'StateChangeConditions', {'SoftCode1', 'ITI_Pre'},...
            'OutputActions', {'GlobalTimerTrig', '000', 'SoftCode', 7});
       
        sma = AddState(sma, 'Name', 'ITI_Pre', ...
            'Timer', S.GUI.ITI_Pre,...
            'StateChangeConditions', {'Tup', 'LED_Onset'},...
            'OutputActions', {'GlobalTimerTrig', '100'});

        sma = AddState(sma, 'Name', 'LED_Onset', ...
            'Timer', 0,...
            'StateChangeConditions', {'GlobalTimer1_Start', 'LED_Puff_ISI'},...
            'OutputActions', {'GlobalTimerTrig', '011'});

        sma = AddState(sma, 'Name', 'LED_Puff_ISI', ...
            'Timer', S.GUI.AirPuff_OnsetDelay,...
            'StateChangeConditions', {'Tup', 'AirPuff'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'AirPuff', ...
            'Timer', S.GUI.AirPuff_Dur,...
            'StateChangeConditions', {'Tup', 'ITI_Post'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'ITI_Post', ...
            'Timer', S.GUI.ITI_Post,...
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', {'GlobalTimerCancel', '011'});         
 
        sma = AddState(sma, 'Name', 'ITI', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', '>exit'},...
            'OutputActions', {'GlobalTimerCancel', '100'}); 

        SendStateMachine(sma); % Send the state matrix to the Bpod device   

        RawEvents = RunStateMachine; % Run the trial and return events

        disp(['Processing Trial Video...']);      

        MEV.processTrialsVideo(vidDur);

        disp(['Trial Video Processed...']);

        if ~isempty(fieldnames(RawEvents)) % If trial data was returned (i.e. if not final trial, interrupted by user)
            BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
            BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
            % m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 1);            
            if useStateTiming
                StateTiming();
            end

            % save FEC session data
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FECRaw = MEV.fecDataRaw;
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FEC = MEV.fecData;
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FECTimes = MEV.fecTimes;    
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.eyeAreaPixels = MEV.arrEyeAreaPixels;
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.totalEllipsePixels = MEV.arrTotalEllipsePixels;            
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FECTrialStartThresh = MEV.arrFECTrialStartThresh;
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.minFur = MEV.minFur; % max eye open threshold

            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.ISI = ISI;
    
            % Update rotary encoder plot
            % might reduce this section to pass
            % BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States, and
            % access them in plot function
            TrialDuration = BpodSystem.Data.TrialEndTimestamp(currentTrial)-BpodSystem.Data.TrialStartTimestamp(currentTrial);
     
            SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file
        end
        % if BpodSystem.Status.Pause == 1
        % 
        % end
        HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
        if BpodSystem.Status.BeingUsed == 0 % If protocol was stopped, exit the loop

            BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session

            MEV.stopTrialsVideo;
            MEV.onGUIClose;
            MEV = [];
            return
        end
    
    end
    
    MEV = [];

    BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session

catch MatlabException
    disp(MatlabException.identifier);
    disp(getReport(MatlabException));

    % err report log file
    % recording error and stack information to file
    t = datetime;
    session_date = 10000*(year(t)-2000) + 100*month(t) + day(t);
    
    % get session file name
    [SessionFilepath, SessionFileName, Ext] = fileparts(BpodSystem.Path.CurrentDataFile);

    CrashFileDir = 'C:\data analysis\behavior\joystick\error logs\';
    CrashFileName = [CrashFileDir, num2str(session_date), '_BPod-matlab_crash_log_', SessionFileName];    

    % make crash log folder if it doesn't already exist
    [status, msg, msgID] = mkdir(CrashFileDir);

    % save workspace variables associated with session
    Data = BpodSystem.Data;
    save(CrashFileName, 'Data');
    % add more workspace vars if needed

    %open file
    fid = fopen([CrashFileName, '.txt'],'a+');

    % write session associated with error
    fprintf(fid,'%s\n', SessionFileName);

    % date
    fprintf(fid,'%s\n', num2str(session_date));

    % rig specs
    fprintf(fid,'%s\n', 'Joystick Rig - Behavior Room');
    % fprintf(fid,'%s\n', computer);
    % fprintf(fid,'%s\n', feature('GetCPU'));
    % fprintf(fid,'%s\n', getenv('NUMBER_OF_PROCESSORS'));
    % fprintf(fid,'%s\n', memory); % add code to print memory struct to
    % file later   

    % write the error to file   
    fprintf(fid,'%s\n',MatlabException.identifier);
    fprintf(fid,'%s\n',MatlabException.message);
    % fprintf(fid,'%s\n',MatlabException.stack);
    % fprintf(fid,'%s\n',MatlabException.cause);
    fprintf(fid,'%s\n',MatlabException.Correction);

    % print stack
    fprintf(fid, '%s', MatlabException.getReport('extended', 'hyperlinks','off'));

    % close file
    fclose(fid);

    % save workspace variables associated with session to file
    

    % disp('Resetting encoder and maestro objects...');
    BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session
    % try
    %     % BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
    %     % BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
    %     % BpodSystem.PluginObjects.R = [];
    % catch ME2
    %     disp(ME2.identifier)
    %     disp(getReport(ME2));
    %     disp('Encoder not initialized.');
    % end
    % M = [];
    try
        MEV.onGUIClose;
        MEV = [];
    catch MatlabException
        disp(MatlabException.identifier);
        disp(getReport(MatlabException));
    end
end
end


