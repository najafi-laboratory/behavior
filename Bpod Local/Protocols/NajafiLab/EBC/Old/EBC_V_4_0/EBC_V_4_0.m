function EBC_V_4_0
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
    

    % Get the selected option from the popup menu
    selectedSD = S.GUI.SleepDeprived; 
    % S.GUI.UseProbeTrials = 1;  % Enable random CS-only probe trials in each block
    
    probeTrialsThisBlock = []; 
    probeIndices = {};
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
    probeTrialsThisBlock = []; 
    probeIndices = {};


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
        isProbeTrial = false;
        % input('Set parameters and press enter to continue >', 's');
        S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
        S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin           



        % switch S.GUI.SleepDeprived
        %     case 1
        %         disp(['Trial',num2str(currentTrial),': Control enabled, No SD']);
        %     case 2
        %         disp(['Trial',num2str(currentTrial),': Post_EBC SD (Control) enabled, SD was done right after EBD']);
        %     case 3
        %         disp(['Trial',num2str(currentTrial),': First SD session enabled, immediately following Post_EBC SD']);
        %     case 4
        %         disp(['Trial',num2str(currentTrial),': Second SD session enabled, subsequent day after SD+1 session']);    
        % 
        % end
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
      
                end


               % ========== Initialize probe trial indices ==========
                if S.GUI.UseProbeTrials
                    probeIndices = {};  % Use a cell array

                        numLeadIn = S.GUI.num_initial_nonprobe_trials_per_block;  % GUI var
                        minTrial = S.GUI.num_warmup_trials + numLeadIn + 1;
                        maxTrial = S.GUI.BlockLength - 10;
                        
                        % Available trials in block
                        numAvailableTrials = maxTrial - minTrial + 1;
                        
                        % How many probe trials in this block?
                        desiredNumProbes = S.GUI.num_probetrials_perBlock;
                
                        % Adjust if block is too short
                        if numAvailableTrials <= 0
                            warning(['Block #' num2str(blockIdx) ' is too short for probes! Skipping probe trials in this block.']);
                            probeTrialsThisBlock = [];  % No probes
                        else
                            numProbes = min(desiredNumProbes, numAvailableTrials);  % Cannot request more probes than available trials
                
                            % Randomly select
                            probeTrialsThisBlock = randsample(minTrial:maxTrial, numProbes);
                        end
                        
                        % Save it
                        % probeIndices{blockIdx} = probeTrialsThisBlock;
                        % 
                        % disp(['probe trials:']);
                        % disp(probeTrialsThisBlock);
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
            % Check if we are at a probe trial
            
            if S.GUI.UseProbeTrials && any(currentTrialInBlock == probeTrialsThisBlock)
                
                    isProbeTrial = true;
                    % disp(['###################################################################################']);
                    % disp(['Probe trial inserted: Trial #' num2str(currentTrial) ' in block #' num2str(currentBlockIndex)]);
                    % disp(['###################################################################################']);
                
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

        fprintf('====================================================\n\n');


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

        sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', S.GUI.LED_Dur, 'OnsetDelay', S.GUI.LED_OnsetDelay,...
            'Channel', 'PWM3', 'PulseWidthByte', 255, 'PulseOffByte', 0,...
            'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0); 


          sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', S.GUI.AirPuff_Dur, ...
            'OnsetDelay', AirPuff_OnsetDelay, 'Channel', 'Valve1', ...
            'OnMessage', 1, 'OffMessage', 0);


        





        sma = SetGlobalTimer(sma, 'TimerID', 3, 'Duration', CamTrigOnDur, 'OnsetDelay', 0, 'Channel', 'BNC1',...
            'OnLevel', 1, 'OffLevel', 0,...
            'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', CamTrigOffDur);         



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

       
  

        if isProbeTrial
          nextStateAfterLED = 'ITI_Post'; % Skip AirPuff
        else
          nextStateAfterLED = 'AirPuff'; % Continue to AirPuff
        end

        % LED Onset State
        sma = AddState(sma, 'Name', 'LED_Onset', ...
          'Timer', 0, ...
          'StateChangeConditions', {'GlobalTimer1_Start', 'LED_Puff_ISI'}, ...
          'OutputActions', {'GlobalTimerTrig', '011', 'SoftCode', 3});

        % LED_Puff_ISI: decides what comes next based on isProbeTrial
        sma = AddState(sma, 'Name', 'LED_Puff_ISI', ...
          'Timer', AirPuff_OnsetDelay, ...
          'StateChangeConditions', {'Tup', nextStateAfterLED}, ...
          'OutputActions', {});

        % AirPuff state (only used if not a probe trial)
        sma = AddState(sma, 'Name', 'AirPuff', ...
          'Timer', S.GUI.AirPuff_Dur, ...
          'StateChangeConditions', {'Tup', 'ITI_Post'}, ...
          'OutputActions', {'SoftCode', 4});

        
        sma = AddState(sma, 'Name', 'ITI_Post', ...
          'Timer', S.GUI.ITI_Post, ...
          'StateChangeConditions', {'Tup', 'ITI'}, ...
          'OutputActions', {'GlobalTimerCancel', '011'});
        sma = AddState(sma, 'Name', 'CheckEyeOpenTimeout', ...
          'Timer', 0, ...
          'StateChangeConditions', {'Tup', 'ITI'}, ...
          'OutputActions', {});
        sma = AddState(sma, 'Name', 'ITI', ...
          'Timer', S.GUI.ITI_Extra, ...
          'StateChangeConditions', {'Tup', 'CamOff'}, ...
          'OutputActions', {});
        sma = AddState(sma, 'Name', 'CamOff', ...
          'Timer', 0, ...
          'StateChangeConditions', {'Tup', '>exit'}, ...
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

            % % CheckEyeOpen Timed Out?
            % wasCheckEyeOpenTimeout = ~isnan(BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.CheckEyeOpenTimeout(1));
            % if wasCheckEyeOpenTimeout
            %     numCheckEyeOpenTimeouts = numCheckEyeOpenTimeouts + 1;
            % end
            
            wasCheckEyeOpenTimeout = ~isnan(BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.CheckEyeOpenTimeout(1));
            if wasCheckEyeOpenTimeout
              numCheckEyeOpenTimeouts = numCheckEyeOpenTimeouts + 1;
              disp(['Skipping trial count increment for Trial #' num2str(currentTrial) ' due to eye closure.']);
            else
              currentTrialInBlock = currentTrialInBlock + 1;
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
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.IsProbeTrial = isProbeTrial; % 

            % Update rotary encoder plot
            % might reduce this section to pass
            % BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States, and
            % access them in plot function
            TrialDuration = BpodSystem.Data.TrialEndTimestamp(currentTrial)-BpodSystem.Data.TrialStartTimestamp(currentTrial);
     
            BpodSystem.Data.ExperimenterInitials = S.GUI.ExperimenterInitials;
            
            % BpodSystem.Data.TrialTypeSequence = S.GUIMeta.TrialTypeSequence.String{S.GUI.TrialTypeSequence};
            % Save both index and label
            BpodSystem.Data.SleepDeprived = S.GUI.SleepDeprived;
            BpodSystem.Data.SleepDeprivedLabel = S.GUIMeta.SleepDeprived.String{S.GUI.SleepDeprived};


            % Show the following information to the Experimenter
            protocol_version = 'EBC_V_4_0';
            % fprintf('*******************************************************\n');

            % fprintf('\bf Bold text\n');  % This will just print "\bf Bold text"
            % Print basic session info

            switch S.GUI.SleepDeprived
                case 1
                    sessionTypeStr = 'Control enabled, No SD';
                case 2
                    sessionTypeStr = 'Post_EBC SD (Control) enabled, SD was done right after EBD';
                case 3
                    sessionTypeStr = 'SD+1 --> First SD session enabled, immediately following Post_EBC SD';
                case 4
                    sessionTypeStr = 'SD+2 --> Second SD session enabled, subsequent day after SD+1 session';
                otherwise
                    sessionTypeStr = 'Unknown';
            end

            % fprintf('Session Type: %s\n', sessionTypeStr);
            fprintf('\n================= SESSION SUMMARY =================\n\n');
            fprintf('Protocol Version: %s\n', protocol_version);
            fprintf('Experimenter: %s\n', S.GUI.ExperimenterInitials);
            fprintf('Session Type: %s\n', sessionTypeStr);
            fprintf('warm_up trial numbers: %d\n', S.GUI.num_warmup_trials);
            fprintf('Probe Trials Enabled: %s\n', string(S.GUI.UseProbeTrials));
            fprintf('probe trial numbers per block: %d\n', S.GUI.num_probetrials_perBlock);
            fprintf('First %d trials per block are non-probes\n', S.GUI.num_initial_nonprobe_trials_per_block);
            
            % fprintf('====================================================\n\n');
            % Session_Level info
            % fprintf('---------------------------------- \n');
            % fprintf('Session info \n');
            
            
            
            % fprintf('First %d trials of each block are fixed as non-probe trials.\n', S.GUI.num_initial_nonprobe_trials_per_block);
            % TrialTypeSequence-specific info
            switch S.GUI.TrialTypeSequence
                case 1  % Single Block
                    fprintf('Single Block Mode: LED/Puff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_SingleBlock);
                case 2  % DoubleBlock_ShortFirst
                    fprintf('DoubleBlock (Short First):\n');
                    fprintf('  Short Block AirPuff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Short);
                    fprintf('  Long Block AirPuff Onset Delay  = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Long);
                case 3  % DoubleBlock_LongFirst
                    fprintf('DoubleBlock (Long First):\n');
                    fprintf('  Long Block AirPuff Onset Delay  = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Long);
                    fprintf('  Short Block AirPuff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Short);
                case 4  % DoubleBlock_RandomFirst
                    fprintf('DoubleBlock (Random Order):\n');
                    fprintf('  Short Block AirPuff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Short);
                    fprintf('  Long Block AirPuff Onset Delay  = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Long);
                otherwise
                    warning('Unknown TrialTypeSequence code: %d\n', S.GUI.TrialTypeSequence);
            end
            % Trial_level info 
            fprintf('---------------------------------- \n');
            fprintf('Trial_level info \n');
            if isProbeTrial
                fprintf('This is a Probe Trial: No Airpuff after LED \n');
            else  
                fprintf('This is a Normal Trial: Airpuff after LED \n');
            end
            fprintf('Current Trial Block Type: %s\n', currentBlockType);
            % fprintf('Current Trial Number: %s\n', ExperimenterTrialInfo.TrialNumber);
            fprintf('ISI Current Trial: %.3f sec\n', ISI); % already in seconds
            fprintf('LED Duration Current Trial: %.3f sec\n', S.GUI.LED_Dur);
            fprintf('AirPuff Duration Current Trial: %.3f sec\n', S.GUI.AirPuff_Dur);
            

            SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file
        end

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

  
   
    BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session

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

