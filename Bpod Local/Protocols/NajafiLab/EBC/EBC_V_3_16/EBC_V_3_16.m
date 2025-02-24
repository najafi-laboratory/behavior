function EBC_V_3_16
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
     
 
    set(BpodSystem.ProtocolFigures.ParameterGUI, 'Position', [9 53 817 548]);
    

    if S.GUI.SleepDeprived
        disp('Sleep deprivation condition is active');
    else
        disp('Sleep deprivation condition is not active');
    end
    
    %% state timing plot
    useStateTiming = true;  % Initialize state timing plot
    if ~verLessThan('matlab','9.5') % StateTiming plot requires MATLAB r2018b or newer
        useStateTiming = true;
        StateTiming();
    end
    
    MEV = EyelidAnalyzer(BpodSystem.GUIData.SubjectName);
    MEV.initPlotLines();

    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_EBC';

    % % Create and start a timer for air puff pulse
    % BpodSystem.Data.AirPuffPulseTimer = timer('TimerFcn',@(x,y)BpodSystem.ToggleOutputState(), 'ExecutionMode', 'fixedSpacing', 'Period', S.GUI.AirPuff_Pulse_Dur);
    BpodSystem.Data.AirPuffPulseTimer = [];
    % if isempty(BpodSystem.AirPuffPulseTimer) || ~isvalid(BpodSystem.AirPuffPulseTimer)
    % 
    %     start(BpodSystem.AirPuffPulseTimer);
    %     disp(['AirPuffPulseTimer Started']);
    % else
    %     start(BpodSystem.AirPuffPulseTimer);
    %     disp(['AirPuffPulseTimer reStarted']);
    % end 
    

    % wait for parameter update and confirm before beginning trial loop
    input('Set parameters and press enter to continue >', 's'); 
    S = BpodParameterGUI('sync', S);
       
    %% init any needed experimenter trial info values
    ExperimenterTrialInfo.TrialNumber = 1;

    %% Main trial loop    

    MaxTrials = 1000;

    MEV.stopPreTrialVideo();
    
    numCheckEyeOpenTimeouts = 0;

    %% Define Warm-Up Trials
    numWarmupTrials = S.GUI.num_warmup_trials; % Get number of warm-up trials from GUI
    isWarmupPhase = true;
   

    
    %% Define block parameters
    % Define block parameters
    
    minBlockLength = S.GUI.BlockLength - S.GUI.Margine;  % Minimum number of trials in a block
    maxBlockLength = S.GUI.BlockLength + S.GUI.Margine;  % Maximum number of trials in a block

    %% **Divide Trials into Blocks (Only for DoubleBlock Mode)**
    
    blocks = [];  % Array to store the length of each block
    
    if S.GUI.TrialTypeSequence == 1  % **SingleBlock Mode**
        % **Do NOT divide into blocks → Use one fixed sequence**
        blocks = MaxTrials - numWarmupTrials;  % **One single block for all trials**
    else
        totalTrials = numWarmupTrials;  % **Start counting from warm-up trials**
        % **Only divide into blocks for DoubleBlock modes**
        while totalTrials < MaxTrials
            currentBlockLength = randi([minBlockLength, maxBlockLength]); % Randomized block length
    
            % Ensure totalTrials doesn't exceed MaxTrials
            if totalTrials + currentBlockLength > MaxTrials
                currentBlockLength = MaxTrials - totalTrials; 
            end
    
            % Add the block to the list of blocks
            blocks = [blocks, currentBlockLength];
    
            % Update total assigned trials
            totalTrials = totalTrials + currentBlockLength;
        end
    end
    %% Initialize Experiment Loop
    currentBlockIndex = 1;
    currentTrialInBlock = 1;

    % Initialize for the experiment loop
    % currentBlockIndex = 1;
    % currentBlockType = 'short'; % Start with a short block
    % S.GUI.AirPuff_OnsetDelay = 0.2;  % 200 ms puff delay for short block
    % use seperate gui parameters (in InitGUI.m) for short and long
    % AirPuff onset delays so that the experimenter can set the delays from
    % the gui
    
    % also use a local variable to store and use the AirPuff_OnsetDelay for the
    % current trial based on the trial type sequence
    % Initialize trial type sequence
    % Initialize Warm-Up Phase
  
    
   
    if isWarmupPhase
        % Warm-up trials logic runs separately before normal trials
        disp('Running Warm-Up Trials...');
    else
        % Initialize Trial Type Sequence for normal trials
        switch S.GUI.TrialTypeSequence

           case 1  % singleBlock → No alternation, use constant ISI
                currentBlockType = 'single';
                AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_SingleBlock;

     
            case 2  % doubleBlock_shortFirst → Start with short delay, then alternate
                currentBlockType = 'short';
                AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
    
            case 3  % doubleBlock_longFirst → Start with long delay, then alternate
                currentBlockType = 'long';
                AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
    
            case 4  % doubleBlock_RandomFirst → Start randomly with short or long
                if rand < 0.5
                    currentBlockType = 'short';
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                else
                    currentBlockType = 'long';
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                end
        end
    end
       % Track the current trial within the block

    for currentTrial = 1:MaxTrials        
        %% sync trial-specific parameters from GUI
      
        % input('Set parameters and press enter to continue >', 's');
        S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
        S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin           

        if S.GUI.SleepDeprived
            disp(['Trial',num2str(currentTrial),': Sleep deprivation enabled']);
        else
            disp(['Trial',num2str(currentTrial),': Sleep deprivation not enabled']);
        end 
        %% Determine trial AirPuff_OnsetDelay
        % These need to be updated prior to
        % MEV.setEventTimes(S.GUI.LED_OnsetDelay, AirPuff_OnsetDelay,
        % S.GUI.ITI_Pre); so that the expected AirPuff onset time is shown
        % for the online experimenter plot
        % Warm-up Phase
        if isWarmupPhase

            % Determine ISI for warm-up trials
            switch S.GUI.TrialTypeSequence
                case 1  % singleBlock
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_SingleBlock;
                    currentBlockType = 'warm_up';
                case {2, 3, 4}  % doubleBlock (short first, long first, or random first)
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                    currentBlockType ='warm_up';
            end


            if currentTrial == numWarmupTrials+1
                isWarmupPhase = false; % End warm-up phase and switch to normal trials
                currentBlockIndex = 1; % Reset block trial counter
                % currentTrialInBlock = 1; 

                if S.GUI.TrialTypeSequence == 1 %Single Block mood
                    currentBlockType = 'single';
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_SingleBlock;
                else %Double Block mood
                    currentBlockIndex = 1; 
                    currentTrialInBlock = 1;
                    currentBlockType = 'short'; %default start type, change based on mood
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                    if S.GUI.TrialTypeSequence == 3
                        currentBlockType = 'long';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                    elseif S.GUI.TrialTypeSequence == 4
                        if rand<0.5
                            currentBlockType = 'short';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                        else
                            currentBlockType = 'long';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                        end
                    end

                    % if strcmp(currentBlockType, 'short')
                    %     AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                    % else
                    %     AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                    % end

                    
                end
            end    

        else
           switch S.GUI.TrialTypeSequence
                case 1  % singleBlock → No alternation, use a constant ISI
                    currentBlockType = 'single';
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_SingleBlock;
                    
            
                case 2  % doubleBlock_shortFirst → Start with short block, then alternate
                    if currentTrialInBlock > blocks(currentBlockIndex)
                        % Alternate between short and long blocks
                        if strcmp(currentBlockType, 'short')
                            currentBlockType = 'long';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                            
                        else
                            currentBlockType = 'short';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                            
                        end
                        % Move to the next block
                        currentBlockIndex = currentBlockIndex + 1;
                        currentTrialInBlock = 1;
                    end
            
                case 3  % doubleBlock_longFirst → Start with long block, then alternate
                    if currentTrialInBlock > blocks(currentBlockIndex)
                        % Alternate between long and short blocks
                        if strcmp(currentBlockType, 'long')
                            currentBlockType = 'short';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                            
                        else
                            currentBlockType = 'long';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                            
                        end
                        % Move to the next block
                        currentBlockIndex = currentBlockIndex + 1;
                        currentTrialInBlock = 1;
                    end
            
               case 4  % doubleBlock_RandomFirst → Start randomly with short or long, then alternate

                    if currentTrialInBlock > blocks(currentBlockIndex)
                        % Randomly choose to start with short or long block
                        if rand < 0.5
                            currentBlockType = 'short';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                            
                        else
                            currentBlockType = 'long';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                            
                        end
                        % Move to the next block
                        currentBlockIndex = currentBlockIndex + 1;
                        currentTrialInBlock = 1;
                    end
            end
        end


        %% Eye Analyzer        
        % move these to trial start function

        MEV.setTrialData();
        MEV.setEventTimes(S.GUI.LED_OnsetDelay, AirPuff_OnsetDelay, S.GUI.ITI_Pre);
        MEV.startTrialsVideo(currentTrial, BpodSystem.GUIData.SubjectName);
        MEV.LEDOnsetTime = 0;
        MEV.AirPuffOnsetTime = 0;
        MEV.plotLEDOnset();
        MEV.plotAirPuffOnset();
        currentTrialNumber_nonTimeout = currentTrial - numCheckEyeOpenTimeouts;
        % MEV.updateTrialNumber_nonTimeout(currentTrialNumber_nonTimeout);

        MEV.eyeOpenAvgWindow = S.GUI.CheckEyeOpenAveragingBaseline;

        %% experimenter info for console display

        ExperimenterTrialInfo.TrialNumber = currentTrial;   % check variable states as field/value struct for experimenter info
        ExperimenterTrialInfo.TrialNumber_CheckEyeOpen = currentTrialNumber_nonTimeout;   % check variable states as field/value struct for experimenter info
        strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
        disp(strExperimenterTrialInfo); 


        %% construct state matrix
    
        sma = NewStateMatrix(); % Assemble state matrix
                       
        % StimDur = max(S.GUI.LED_Dur, S.GUI.AirPuff_OnsetDelay + S.GUI.AirPuff_Dur);
        % Set timer pulses for 250 fps video trigger
        CamPeriod = 0.004;
        fps = 1/CamPeriod;
        MEV.ProtoCamPeriod = CamPeriod;
        CamTrigOnDur = 0.0005;
        CamTrigOffDur = CamPeriod - CamTrigOnDur; %0.0025 - CamTrigOnDur;

        LED_offset = S.GUI.LED_OnsetDelay + S.GUI.LED_Dur;
        ISI = AirPuff_OnsetDelay - LED_offset;
                  
        if ISI <=0  % classical conditioning: no gap between led offset and puff onset
	        LED_puff = S.GUI.LED_Dur;

        else % trace conditioning: puff onset happens after led offset
	        LED_puff = S.GUI.LED_Dur + ISI + S.GUI.AirPuff_Dur;
            
        end

        vidSecondsPreLEDOnsetKeep = min(1, S.GUI.ITI_Pre);
        vidSecondsPostLEDOnsetKeep = LED_puff + min(3, S.GUI.ITI_Post);

        vidDurKeep = vidSecondsPreLEDOnsetKeep + S.GUI.LED_OnsetDelay + vidSecondsPostLEDOnsetKeep; % unless LED dur is later than puff offset
        vidDur = S.GUI.ITI_Pre + S.GUI.LED_OnsetDelay + LED_puff + S.GUI.ITI_Post + S.GUI.ITI_Extra; % unless LED dur is later than puff offset
        numFramesVidKeep = round(fps * vidDurKeep);
        numFramesVid = round(fps * vidDur);
        numFramesITI = round(fps * S.GUI.ITI_Extra);
        % numFramesPreLEDOnset = fps * vidSecondsPreLEDOnset;
        % numFramesPostLEDOnset =  fps * vidSecondsPostLEDOnset;

        % rig-specific code here
        % switch BpodSystem.Data.RigName
        %     case 'ImagingRig'
        %         % rig-specific code here         
        % end


        % LED Timer
        % LED Timer generated using behavior port 1 Pulse Width Modulation 
        % pin (PWM).  Starts after S.GUI.LED_OnsetDelay, on during
        % S.GUI.LED_Dur.
        sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', S.GUI.LED_Dur, 'OnsetDelay', S.GUI.LED_OnsetDelay,...
            'Channel', 'PWM3', 'PulseWidthByte', 255, 'PulseOffByte', 0,...
            'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0); 

        
        % Air Puff Timer
        % Air Puff Timer uses behavior port 1 12V valve pin.
        sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', S.GUI.AirPuff_Dur, 'OnsetDelay', AirPuff_OnsetDelay, 'Channel', 'Valve1',...
                     'OnMessage', 1, 'OffMessage', 0); 
               

        % t = S.GUI.ITI_Pre + S.GUI.LED_OnsetDelay + S.GUI.LED_Dur + 
        % fCam = 400;
        % T = 1 / fCam;
        % Dur = T / 2;




        % Cam Trigger
        % Camera Trigger generated using BNC1 output.  400Hz signal.
        sma = SetGlobalTimer(sma, 'TimerID', 3, 'Duration', CamTrigOnDur, 'OnsetDelay', 0, 'Channel', 'BNC1',...
            'OnLevel', 1, 'OffLevel', 0,...
            'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', CamTrigOffDur);         

        % try
        %     MEV.frame = getsnapshot(MEV.vid);
        %     rgbFrame = double(cat(3, MEV.frame, MEV.frame, MEV.frame)) / 255; % Convert to RGB by replicating the single channel, Normalize to [0, 1] double precision
        %     set(MEV.imgOrigHandle, 'CData', rgbFrame);
        % catch MatlabException
        %     disp(MatlabException.identifier);
        %     disp(getReport(MatlabException));
        % end
        
        % sma = AddState(sma, 'Name', 'CheckEyeOpen', ...
        %     'Timer', 0,...
        %     'StateChangeConditions', {'SoftCode1', 'Start'},...
        %     'OutputActions', {'GlobalTimerTrig', '100', 'SoftCode', 1});
        % 
        % sma = AddState(sma, 'Name', 'Start', ...
        %     'Timer', 0,...
        %     'StateChangeConditions', {'SoftCode1', 'ITI_Pre'},...
        %     'OutputActions', {'GlobalTimerCancel', '100', 'SoftCode', 2});
        % 
        % sma = AddState(sma, 'Name', 'ITI_Pre', ...
        %     'Timer', S.GUI.ITI_Pre,...
        %     'StateChangeConditions', {'Tup', 'LED_Onset'},...
        %     'OutputActions', {'GlobalTimerTrig', '100'});

        sma = AddState(sma, 'Name', 'Start', ...
            'Timer', 0,...
            'StateChangeConditions', {'SoftCode1', 'ITI_Pre'},...
            'OutputActions', {'GlobalTimerTrig', '100', 'SoftCode', 1});
        
        sma = AddState(sma, 'Name', 'ITI_Pre', ...
            'Timer', S.GUI.ITI_Pre,...
            'StateChangeConditions', {'Tup', 'CheckEyeOpen'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'CheckEyeOpen', ...
            'Timer', 0,...
            'StateChangeConditions', {'SoftCode1', 'LED_Onset', 'SoftCode2', 'CheckEyeOpenTimeout'},...
            'OutputActions', {'SoftCode', 2});

        sma = AddState(sma, 'Name', 'LED_Onset', ...
            'Timer', 0,...
            'StateChangeConditions', {'GlobalTimer1_Start', 'LED_Puff_ISI'},...
            'OutputActions', {'GlobalTimerTrig', '011', 'SoftCode', 3});

        sma = AddState(sma, 'Name', 'LED_Puff_ISI', ...
            'Timer', AirPuff_OnsetDelay,...
            'StateChangeConditions', {'Tup', 'AirPuff'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'AirPuff', ...
            'Timer', S.GUI.AirPuff_Dur,...
            'StateChangeConditions', {'Tup', 'ITI_Post'},...
            'OutputActions', {'SoftCode', 4});

        sma = AddState(sma, 'Name', 'ITI_Post', ...
            'Timer', S.GUI.ITI_Post,...
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', {'GlobalTimerCancel', '011'});         
 
        sma = AddState(sma, 'Name', 'CheckEyeOpenTimeout', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'ITI', ...
            'Timer', S.GUI.ITI_Extra,...
            'StateChangeConditions', {'Tup', 'CamOff'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'CamOff', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', '>exit'},...
            'OutputActions', {'GlobalTimerCancel', '100'});        

        SendStateMachine(sma); % Send the state matrix to the Bpod device   

        RawEvents = RunStateMachine; % Run the trial and return events

        disp(['Processing Trial Video...']);      


        % MEV.processTrialsVideo(numFramesVid, numFramesITI, wasCheckEyeOpenTimeout);
        MEV.processTrialsVideo(numFramesVid, numFramesITI, numFramesVidKeep);

        disp(['Trial Video Processed...']);


        if ~isempty(fieldnames(RawEvents)) % If trial data was returned (i.e. if not final trial, interrupted by user)
            BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
            BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
            % m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 1);            
            if useStateTiming
                StateTiming();
            end

            % CheckEyeOpen Timed Out?
            wasCheckEyeOpenTimeout = ~isnan(BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.CheckEyeOpenTimeout(1));
            if wasCheckEyeOpenTimeout
                numCheckEyeOpenTimeouts = numCheckEyeOpenTimeouts + 1;
            end


            

            % save FEC session data
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FECRaw = MEV.fecDataRaw;
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FEC = MEV.fecData;
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FECTimes = MEV.fecTimes;    
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.eyeAreaPixels = MEV.arrEyeAreaPixels;
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.totalEllipsePixels = MEV.arrTotalEllipsePixels;            
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.FECTrialStartThresh = MEV.arrFECTrialStartThresh;
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.minFur = MEV.minFur; % max eye open threshold

            % save other session data
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.ISI = ISI; % ISI for current trial
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.AirPuff_OnsetDelay = AirPuff_OnsetDelay; % air puff onset delay for current trial
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.BlockType = currentBlockType; % block type for current trial
   
            % Update rotary encoder plot
            % might reduce this section to pass
            % BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States, and
            % access them in plot function
            TrialDuration = BpodSystem.Data.TrialEndTimestamp(currentTrial)-BpodSystem.Data.TrialStartTimestamp(currentTrial);
     
            BpodSystem.Data.ExperimenterInitials = S.GUI.ExperimenterInitials;
            BpodSystem.Data.SleepDeprived = S.GUI.SleepDeprived;

             %%Show the following information to the Experimenter
            protocol_version = 'EBC_V_3_16';
            % Example protocol version
            % Print the information to the console
            fprintf('Experimenter Initials: %s\n', S.GUI.ExperimenterInitials);
            fprintf('Protocol Version: %s\n', protocol_version);
            fprintf('Block Type: %s\n', currentBlockType);
            fprintf('ISI: %.3f sec\n', ISI); % already in seconds
            % fprintf('ISI: %.3f ms\n', ISI*1000); % where ISI = AirPuff_OnsetDelay - LED_offset;
            fprintf('LED Duration: %.3f sec\n', S.GUI.LED_Dur);
            fprintf('AirPuff Duration: %.3f sec\n', S.GUI.AirPuff_Dur);

            fprintf('warm_up trial numbers: %d\n', S.GUI.num_warmup_trials);
            
            fprintf('doubleBlock_shortFirst LED/Puff Onset Delay: %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Short);
            fprintf('doubleBlock_longFirst LED/Puff Onset Delay: %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Long);
            fprintf('Single Block LED/Puff Onset Delay: %.3f sec\n', S.GUI.AirPuff_OnsetDelay_SingleBlock);



 
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

            if ~isempty(BpodSystem.Data.AirPuffPulseTimer) && isvalid(BpodSystem.Data.AirPuffPulseTimer)
                stop(BpodSystem.Data.AirPuffPulseTimer);
                delete(BpodSystem.Data.AirPuffPulseTimer);
                BpodSystem.Data.AirPuffPulseTimer = [];
            end    

            return
        end
    % Increment trial counter within the block
    currentTrialInBlock = currentTrialInBlock + 1;  % Increment the trial count for the current block
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
    %  %%Show the following information to the Experimenter
    %  protocol_version = 'EBC_V_3_11';
    % % Example protocol version
    % % Print the information to the console
    % fprintf('Protocol Version: %s\n', protocol_version);
    % fprintf('Short Block LED Duration: %d ms\n', S.GUI.LED_Dur_Short);
    % fprintf('Short Block Puff Duration: %d ms\n', S.GUI.AirPuff_Dur);
    % fprintf('Short Block LED/Puff Onset Delay: %d ms\n', S.GUI.AirPuff_OnsetDelay_Short);
    % fprintf('Long Block LED Duration: %d ms\n', S.GUI.LED_Dur_Long);
    % fprintf('Long Block Puff Duration: %d ms\n', S.GUI.AirPuff_Dur);
    % fprintf('Long Block LED/Puff Onset Delay: %d ms\n', S.GUI.AirPuff_OnsetDelay_Long);

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

% match rig ID to computer name for rig-specific settings
% (features/timing/servos/etc)
function SetRigID(BpodSystem)
    BpodSystem.Data.ComputerHostName = getenv('COMPUTERNAME');
    BpodSystem.Data.RigName = '';
    switch BpodSystem.Data.ComputerHostName
        case 'COS-3A11406'
            BpodSystem.Data.RigName = 'ImagingRig';
        case 'COS-3A11427'
            BpodSystem.Data.RigName = 'JoystickRig';
        case 'COS-3A17904'
            BpodSystem.Data.RigName = 'JoystickRig2';
    end
end

