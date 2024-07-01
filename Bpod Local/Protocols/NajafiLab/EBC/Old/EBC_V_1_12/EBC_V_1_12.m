function EBC_V_1_12
try
    global BpodSystem
    global S
    % global M
    global MEV
    
    EnableOpto         = 1;
    updateGUIPos       = 0;
    exitProto          = 0;
    
    %% Import scriptsBpod
    
    m_Plotter = Plotter;
    m_InitGUI = InitGUI;
    m_TrialConfig = TrialConfig;
    m_Opto = OptoConfig(EnableOpto);
    
    %% Turn off Bpod LEDs
    
    % This code will disable the state machine status LED
    BpodSystem.setStatusLED(0);

    % get matlab version
    v_info = version;
    BpodSystem.Data.MatVer = version;
    
  
    %% Define parameters
    [S] = m_InitGUI.SetParams(BpodSystem);
    
  

    %% Initialize plots
    
    % Press Outcome Plot
    % BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
    % BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [918 808 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off'); 
    % BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
    % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'init',(numTrialTypes+1)-TrialTypes);
    % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'init',TrialTypes);
    % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes, OptoTrialTypes);
    BpodParameterGUI('init', S); % Initialize parameter GUI plugin
    % BpodSystem.ProtocolFigures.ParameterGUI.Scrollable

    % BpodSystem.ProtocolFigures.ParameterGUI.Position = [-1026 51 2324 980];
    % BpodSystem.ProtocolFigures.ParameterGUI.Position = [9 53 1617 708];
    BpodSystem.ProtocolFigures.ParameterGUI.Position = [9 53 442 185];
     
 
    set(BpodSystem.ProtocolFigures.ParameterGUI, 'Position', [9 53 697 539]);
    
    %% state timing plot
    useStateTiming = true;  % Initialize state timing plot
    if ~verLessThan('matlab','9.5') % StateTiming plot requires MATLAB r2018b or newer
        useStateTiming = true;
        StateTiming();
    end
    
    MEV = EyelidAnalyzer();
    % MEV.SubjectName = BpodSystem.GUIData.SubjectName;
    MEV.startGUIVideo();
    % if ~MEV.startVideo
    %     MatExc = MException('Set camera FPS to 30 fps.');
    %     throw(MatExc);
    % end

    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_EBC';

    % wait for parameter update and confirm before beginning trial loop
    input('Set parameters and press enter to continue >', 's'); 
    S = BpodParameterGUI('sync', S);
       
    %% init any needed experimenter trial info values
    ExperimenterTrialInfo.TrialNumber = 0;

    %% Main trial loop    

    MaxTrials = 1000;

    MEV.stopGUIVideo;
    % 
    MEV.connectVideoTrial(BpodSystem.GUIData.SubjectName);
    % MEV.startVideoTrial;
    % MEV.startVideoTrial;

    % MEV.stopVideoTrial;

    for currentTrial = 1:MaxTrials
       
        ExperimenterTrialInfo.TrialNumber = currentTrial;   % check variable states as field/value struct for experimenter info
    
        % wait for parameter update and confirm before beginning trial loop
        % input('Set parameters and press enter to continue >', 's'); 

        % MEV.startVideoTrial;

        %% sync trial-specific parameters from GUI
      
        % input('Set parameters and press enter to continue >', 's');
        S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
        S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin    
        

        strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
        disp(strExperimenterTrialInfo);          
    
        %% construct state matrix
    
        sma = NewStateMatrix(); % Assemble state matrix
               
        OptoNumPulses = S.GUI.NumPulses;
        OptoPulseDur = S.GUI.PulseDur;
        IPI = S.GUI.IPI;

        OptoTimerDuration = (OptoPulseDur + IPI) * OptoNumPulses;

        % S.GUI.ITI_Pre = 0.5;
        % S.GUI.ITI_Post = 1;
        % 
        % S.GUI.LED_OnsetDelay = 0;
        % S.GUI.LED_Dur = 0.5;
        % 
        % S.GUI.AirPuff_Dur = 0.02;
        S.GUI.AirPuff_OnsetDelay = S.GUI.LED_Dur - S.GUI.AirPuff_Dur;
        


        StimDur = max(S.GUI.LED_Dur, S.GUI.AirPuff_OnsetDelay + S.GUI.AirPuff_Dur);

        % MEV.startVideoTrial;

        % LED Timer
        sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', S.GUI.LED_Dur, 'OnsetDelay', S.GUI.LED_OnsetDelay,...
            'Channel', 'PWM1', 'PulseWidthByte', 255, 'PulseOffByte', 0,...
            'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0); 

        % sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', 0.5, 'OnsetDelay', 0.5,...
        %              'Channel', 'PWM2', 'PulseWidthByte', 255, 'PulseOffByte', 0,...
        %              'Loop', 1, 'SendGlobalTimerEvents', 1, 'LoopInterval', 0.5); 

        % Air Puff Timer
        sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', S.GUI.AirPuff_Dur, 'OnsetDelay', S.GUI.AirPuff_OnsetDelay, 'Channel', 'Valve1',...
                     'OnMessage', 1, 'OffMessage', 0); 
       
        camF = 470;
        camT = 1/camF;
        camTrigOn = camT/2;
        camTrigOff = camTrigOn;

        % Cam Trigger
        sma = SetGlobalTimer(sma, 'TimerID', 3, 'Duration', 0.002, 'OnsetDelay', 0, 'Channel', 'BNC1',...
            'OnLevel', 1, 'OffLevel', 0,...
            'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', camTrigOff); 
        %'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0.031);

        MEV.startVideoTrial;
        % OptoFramesPerPulse = 3;

        % Pulse Sequence Counter - Run trial for a param-defined number of
        % opto pulses
        % CounterNumber = 1;
        % TargetEventName = 'BNC1High';
        % Threshold = OptoNumPulses * OptoFramesPerPulse;
        
        % sma = SetGlobalCounter(sma, CounterNumber, TargetEventName, Threshold);

        % sma = AddState(sma, 'Name', 'Trigger', ...
        %     'Timer', 0.002,...
        %     'StateChangeConditions', {'Tup', 'CheckEyeOpen'},...
        %     'OutputActions', {'BNC1', 1});
        % 
        % sma = AddState(sma, 'Name', 'CheckEyeOpen', ...
        %     'Timer', 0,...
        %     'StateChangeConditions', {'SoftCode1', 'Trigger', 'SoftCode2', 'Start'},...
        %     'OutputActions', {'SoftCode', 9});

        sma = AddState(sma, 'Name', 'Start', ...
            'Timer', 0,...
            'StateChangeConditions', {'SoftCode1', 'ITI_Pre'},...
            'OutputActions', {'GlobalTimerTrig', '100', 'SoftCode', 7});
        % 'StateChangeConditions', {'SoftCode1', 'ITI_Pre'},...
            % 'OutputActions', {'GlobalTimerTrig', '100', 'SoftCode', 7});
        

        sma = AddState(sma, 'Name', 'ITI_Pre', ...
            'Timer', S.GUI.ITI_Pre,...
            'StateChangeConditions', {'Tup', 'Stimulus'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'Stimulus', ...
            'Timer', StimDur,...
            'StateChangeConditions', {'Tup', 'ITI_Post'},...
            'OutputActions', {'GlobalTimerTrig', '011'});

        % 'StateChangeConditions', {'GlobalTimer2_End', 'ITI_Post'},...

        sma = AddState(sma, 'Name', 'ITI_Post', ...
            'Timer', S.GUI.ITI_Post,...
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', {'GlobalTimerCancel', '111'});         
 
        sma = AddState(sma, 'Name', 'ITI', ...
            'Timer', 0,...
            'StateChangeConditions', {'SoftCode1', '>exit'},...
            'OutputActions', {'GlobalTimerCancel', '100', 'SoftCode', 8}); 

        SendStateMachine(sma); % Send the state matrix to the Bpod device   
        % pause(2);

        RawEvents = RunStateMachine; % Run the trial and return events

        % MEV.stopVideoTrial;
        disp(['vid proc']);
        % MEV.processVideoTrial(currentTrial, BpodSystem.GUIData.SubjectName);

        BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FEC = MEV.fecData;
        BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FECTimes = MEV.fecTimes;

        MEV.fecData = [];
        MEV.fecTimes = [];


        if ~isempty(fieldnames(RawEvents)) % If trial data was returned (i.e. if not final trial, interrupted by user)
            BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
            BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
            % BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial); % Adds the trial type of the current trial to data
            % m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 1);
            if useStateTiming
                StateTiming();
            end
    
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
            % BpodSystem.PluginObjects.V = [];
            BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session
            % BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
            % BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
            % BpodSystem.PluginObjects.R = [];      
            % M = [];
            MEV.stopVideoTrial;
            MEV.onGUIClose;
            MEV = [];
            return
        end
    
    end
    
    MEV = [];
    % BpodSystem.PluginObjects.V = [];
    BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session
    % BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
    % BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
    % BpodSystem.PluginObjects.R = [];
    % M = [];
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


