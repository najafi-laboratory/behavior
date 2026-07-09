function EBC_Opto_V_5_0
try
    global BpodSystem
    global S
    global MEV
    
    
    %% init encoder object
    BpodSystem.PluginObjects.R = struct;

    %% Assert Rotary Encoder modules are present + USB-paired (via USB button on console GUI)
    disp('Connecting Encoder...');
    BpodSystem.assertModule('RotaryEncoder', 1); % The second argument [1 1] indicates that both HiFi and RotaryEncoder must be paired with their respective USB serial ports
    BpodSystem.PluginObjects.R = RotaryEncoderModule(BpodSystem.ModuleUSB.RotaryEncoder1);     

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
    BpodSystem.ProtocolFigures.ParameterGUI.Position = [6 169 1209 616];
     
 
    set(BpodSystem.ProtocolFigures.ParameterGUI, 'Position', [6 169 1209 616]);
    

    % Get the selected option from the popup menu
    selectedSD = S.GUI.SleepDeprived; 
    % S.GUI.UseProbeTrials = 1;  % Enable random CS-only probe trials in each block
    
    probeTrialsThisBlock = []; 
    probeIndices = {};

    %% Last Trial encoder plot (an online plot included in the protocol folder)
    BpodSystem.ProtocolFigures.EncoderPlotFig = figure('Position', [-2 47 1500 600],'name','Encoder plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
    BpodSystem.GUIHandles.EncoderAxes = axes('Position', [.15 .15 .8 .8]);
    LastTrialEncoderPlot(BpodSystem.GUIHandles.EncoderAxes, 'init');   

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


    % Set parameters based on selected training stage
    switch S.GUI.TrainingStage
        case 1  % Naive
            % Force to fixed, simple setting
            S.GUI.TrialTypeSequence = 1;  % singleBlock
            S.GUI.BlockType = 2;          % long
            % probetrials_percentage_perBlock = 0;
        case 2  % Well-trained
            % Use whatever TrialTypeSequence is already chosen in the GUI (2–6)
            % No need to override or modify it here
            % BlockType and other parameters follow GUI or protocol logic
    end

    % if S.GUI.TrialTypeSequence == 1  % **SingleBlock Mode**
    %     % **Do NOT divide into blocks → Use one fixed sequence**
    %     blocks = MaxTrials - numWarmupTrials;  % **One single block for all trials**
    % else
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
    % end
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

            case 2  % SingleTransition short to long → Start with short, then the rest of the blocks are long
                currentBlockType = 'short';
                AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;

            case 3  % SingleTransition long to short → Start with long, then the rest of the blocks are short
                currentBlockType = 'long';
                AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;                
     
            case 4  % doubleBlock_shortFirst → Start with short delay, then alternate
                currentBlockType = 'short';
                AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
    
            case 5  % doubleBlock_longFirst → Start with long delay, then alternate
                currentBlockType = 'long';
                AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
    
            case 6  % doubleBlock_RandomFirst → Start randomly with short or long
                if rand < 0.5
                    currentBlockType = 'short';
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                else
                    currentBlockType = 'long';
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                end
       end
    end

    %Track the current trial within the block
    BlockTypes = repmat({''}, 1, MaxTrials);   %  'short'/'long'/'single'/'warm_up' string-safe
    probeTrialsThisBlock = [];
    probeTrialsByBlock = cell(1, numel(blocks));   % block-relative lists
    optoTrialsByBlock= cell(1, numel(blocks)); 
    optoTrialsThisBlock  = [];                 % absolute trial numbers (current block only)
    % isTransitionOptoBlock = false;
    
    TransitionCount    = 0;                    % counts qualifying short→long transitions seen so far
    isTransitionOptoBlock = false;             % true if this block is an opto-at-transition block
    OptoTrialNums     = [];
    allowThisTransition = false;
    
    %% Setup rotary encoder module
    BpodSystem.PluginObjects.R.setPosition(0); % Set the current position to equal 0
    BpodSystem.PluginObjects.R.wrapPoint = 0; % 0 wrap point enables continuously tracking position up to 32 rotations (NOTE: 16 rotations usually, since we start at 'half-way' of zero degrees rather than =/- 2880 deg)
    BpodSystem.PluginObjects.R.wrapMode = 'unipolar'; % 'bipolar' (position wraps to negative value) or 'unipolar' (wraps to 0)  = 0;
    % BpodSystem.PluginObjects.R.thresholds = S.GUI.Threshold;
    BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % If on, rotary encoder module will send threshold events to state machine
    BpodSystem.PluginObjects.R.startUSBStream;


    %% sync trial-specific parameters from GUI
    % main loop for trials
    
   for currentTrial = 1:MaxTrials
       
        isProbeTrial = false;
        isOptoTrial = false;
        qualifies = false;
        isTransitionOptoBlock = false;
        allowThisTransition = false;
       

        % input('Set parameters and press enter to continue >', 's');

        S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
        S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin

        % Warm-up Phase
        if isWarmupPhase
            % Determine ISI for warm-up trials
            switch S.GUI.TrialTypeSequence
                case 1  % singleBlock
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_SingleBlock;
                    currentBlockType = 'warm_up';
                case {2, 3, 4, 5, 6}  % doubleBlock (short first, long first, or random first)
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                    currentBlockType ='warm_up';
            end

            % --------------------------------------------------------------
            % PHASE 1: First block immediatly after warmups is here
            % --------------------------------------------------------------
            if currentTrial == numWarmupTrials
                isWarmupPhase = false; % End warm-up phase
                currentBlockIndex = 1; % Reset block trial counter
                switch S.GUI.TrialTypeSequence
                    case 1  % Single block
                        currentBlockType = 'single';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_SingleBlock;
                    case 2  % singleTransition_short_to_long
                        currentBlockType = 'short';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                    case 3  % singleTransition_long_to_short
                        currentBlockType = 'long';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                    case 4  % doubleBlock_shortFirst
                        currentBlockType = 'short';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                        currentTrialInBlock = 1;
                    case 5  % doubleBlock_longFirst
                        currentBlockType = 'long';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                        currentTrialInBlock = 1;
                    case 6  % doubleBlock_RandomFirst
                        if rand < 0.5
                            currentBlockType = 'short';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                        else
                            currentBlockType = 'long';
                            AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                        end
                        currentTrialInBlock = 1;
                end
                BlockTypes{currentTrial} = currentBlockType;
               % ========== Initialize probe trial indices ==========


                if S.GUI.UseProbeTrials == 1
                  numLeadIn = S.GUI.num_initial_nonprobe_trials_perBlock;
                  if currentBlockIndex == 1
                    blockStartTrial = 1+ numWarmupTrials;
                  else
                    blockStartTrial = sum(blocks(1:currentBlockIndex - 1)) + 1;
                  end
                  blockEndTrial = blockStartTrial + blocks(currentBlockIndex) - 1;
                  minTrial = blockStartTrial + numLeadIn;
                  maxTrial = blockEndTrial - 2;  % exclude last trial from probes
                  probe_candidate_trials = minTrial:maxTrial;
                  numCandidateTrials = length(probe_candidate_trials);
                  % percentage-based target
                  probe_fraction = S.GUI.probetrials_percentage_perBlock / 100;
                  numProbes = round(probe_fraction * numCandidateTrials);
                  % --- Random selection with minimum spacing (bounded) ---
                  gap = S.GUI.ProbeMinSeparation;   % e.g., 3 -> need >= 3 non-probes between probes
                  sep = gap + 1;            % index separation
                  selected = [];
                  % keep picking until we hit numProbes (target) or run out of candidates
                  while numel(selected) < numProbes && ~isempty(probe_candidate_trials)
                    % pick one candidate at random
                    t = probe_candidate_trials(randi(numel(probe_candidate_trials)));
                    selected(end+1) = t; %#ok<AGROW>
                    % remove any candidates too close (< sep) to the chosen one
                    probe_candidate_trials = probe_candidate_trials(abs(probe_candidate_trials - t) >= sep);
                   end

                  probeTrialsThisBlock = sort(selected);
                  probeTrialsByBlock{currentBlockIndex} = probeTrialsThisBlock;

                  fprintf('Block #%d: %d probe trials selected from %d–%d (excl first %d; excl last %d); min gap=%d\n', ...
                    currentBlockIndex, numel(probeTrialsThisBlock), minTrial, maxTrial, numLeadIn, blockEndTrial, S.GUI.ProbeMinSeparation);
                  fprintf('Probe trial indices: %s\n', mat2str(probeTrialsThisBlock));
                end


               % ========== Initialize OPTO selection ==========
               if S.GUI.OptoEnabled
                    switch S.GUI.OptoSessionType
                        case 1  % RandomTrial
                              
                              if currentBlockIndex == 1
                                blockStartTrial = 1+ numWarmupTrials;
                              else
                                blockStartTrial = sum(blocks(1:currentBlockIndex - 1)) + 1;
                              end
                              blockEndTrial = blockStartTrial + blocks(currentBlockIndex) - 1;
                              % minTrial = blockStartTrial;
                              % maxTrial = blockEndTrial - 2;  % exclude last trial from probes
                              % opto_candidate_trials = minTrial:maxTrial;
                              blockTrialRange = blockStartTrial:blockEndTrial;

                              numOptoCandidateTrials = length(blockTrialRange);

                              fprintf('Opto Candidate numbers : %d\n', numOptoCandidateTrials);
                              % percentage-based target
                              opto_fraction = S.GUI.OptoFraction;
                              numOpto = round(opto_fraction * numOptoCandidateTrials);
                              fprintf('Opto trial numbers : %d\n', numOpto );
                              % --- Random selection with minimum spacing (bounded) ---
                              Optogap = S.GUI.OptoMinSeparation;   % e.g., 3 -> need >= 3 non-probes between probes
                              Optosep = Optogap + 1;            % index separation
                              Optoselected = [];
                              % keep picking until we hit numProbes (target) or run out of candidates
                              while numel(Optoselected) < numOpto && ~isempty(blockTrialRange)
                                % pick one candidate at random
                                o = blockTrialRange(randi(numel(blockTrialRange)));
                                Optoselected(end+1) = o; %#ok<AGROW>
                                % remove any candidates too close (< sep) to the chosen one
                                blockTrialRange = blockTrialRange(abs(blockTrialRange - o) >= Optosep);
                               end
            
                              optoTrialsThisBlock = sort(Optoselected);
                              optoTrialsByBlock{currentBlockIndex} = optoTrialsThisBlock;
            
                              fprintf('Opto trial indices: %s\n', mat2str(optoTrialsThisBlock));
                         case 2  % ProbeTrialOpto (no pre-assign; set per-trial)
                            optoTrialsThisBlock = probeTrialsThisBlock; 
                            optoTrialsByBlock{currentBlockIndex} = optoTrialsThisBlock;
                            fprintf('[Block %d] ProbeTrialOpto mode (opto follows probe).\n', currentBlockIndex);

                         case 3  % BlockTransition: first N of long after short
                            fprintf('This is first block, No Transition OPTO trials!!\n');

                   end
               end
            end
     else
      % --------------------------------------------
      % PHASE 2: Rest of the blocks Main training 
      % --------------------------------------------

        switch S.GUI.TrialTypeSequence
            case 1 % singleBlock → fixed ISI
                currentBlockType = 'single';
                AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_SingleBlock;
                if currentTrialInBlock > blocks(currentBlockIndex)
                    currentBlockIndex = currentBlockIndex + 1;
                    currentTrialInBlock = 1;
                end
            case 2 % singleTransition_short_to_long
                if currentTrialInBlock > blocks(1)
                    currentBlockType = 'long';
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                end
                if currentTrialInBlock > blocks(currentBlockIndex)
                    currentBlockIndex = currentBlockIndex + 1;
                    currentTrialInBlock = 1;
                end

            case 3 % singleTransition_long_to_short
                if currentTrialInBlock > blocks(1)
                    currentBlockType = 'short';
                    AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                end
                if currentTrialInBlock > blocks(currentBlockIndex)
                    currentBlockIndex = currentBlockIndex + 1;
                    currentTrialInBlock = 1;
                end

            case {4, 5} % doubleBlock alternating (shortFirst or longFirst)
                if currentTrialInBlock > blocks(currentBlockIndex)
                    % Alternate block type
                    if strcmp(currentBlockType, 'short')
                        currentBlockType = 'long';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                    else
                        currentBlockType = 'short';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                    end
                    currentBlockIndex = currentBlockIndex + 1;
                    currentTrialInBlock = 1;
                end
            case 6 % doubleBlock_RandomFirst → Random type at every block
                if currentTrialInBlock > blocks(currentBlockIndex)
                    if rand < 0.5
                        currentBlockType = 'short';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Short;
                    else
                        currentBlockType = 'long';
                        AirPuff_OnsetDelay = S.GUI.AirPuff_OnsetDelay_Long;
                    end
                    currentBlockIndex = currentBlockIndex + 1;
                    currentTrialInBlock = 1;
                end
        end
     BlockTypes{currentTrial} = currentBlockType;

    % ---------- BEGINNING OF BLOCK: select probe & opto trials ----------
    if currentTrialInBlock == 1  
              if S.GUI.UseProbeTrials == 1
                  numLeadIn = S.GUI.num_initial_nonprobe_trials_perBlock;
                  if currentBlockIndex == 1
                    blockStartTrial = 1;
                  else
                    blockStartTrial = sum(blocks(1:currentBlockIndex - 1)) + 1;
                  end
                  blockEndTrial = blockStartTrial + blocks(currentBlockIndex) - 1;
                  if strcmp(currentBlockType, 'single')
                    minTrial = blockStartTrial ;
                  else
                    minTrial = blockStartTrial+numLeadIn ;
                  end    
                  maxTrial = blockEndTrial - 2;  % exclude last trial from probes
                  probe_candidate_trials = minTrial:maxTrial;
                  numCandidateTrials = length(probe_candidate_trials);
                  % percentage-based target
                  probe_fraction = S.GUI.probetrials_percentage_perBlock / 100;
                  numProbes = round(probe_fraction * numCandidateTrials);
                  % --- Random selection with minimum spacing (bounded) ---
                  gap = S.GUI.ProbeMinSeparation;   % e.g., 3 -> need >= 3 non-probes between probes
                  sep = gap + 1;            % index separation
                  selected = [];
                  % keep picking until we hit numProbes (target) or run out of candidates
                  while numel(selected) < numProbes && ~isempty(probe_candidate_trials)
                    % pick one candidate at random
                    t = probe_candidate_trials(randi(numel(probe_candidate_trials)));
                    selected(end+1) = t; %#ok<AGROW>
                    % remove any candidates too close (< sep) to the chosen one
                    probe_candidate_trials = probe_candidate_trials(abs(probe_candidate_trials - t) >= sep);
                   end

                  probeTrialsThisBlock = sort(selected);
                  probeTrialsByBlock{currentBlockIndex} = probeTrialsThisBlock;

                  fprintf('Block #%d: %d probe trials selected from %d–%d (excl first %d; excl last %d); min gap=%d\n', ...
                    currentBlockIndex, numel(probeTrialsThisBlock), minTrial, maxTrial, numLeadIn, blockEndTrial, S.GUI.ProbeMinSeparation);
                  fprintf('Probe trial indices: %s\n', mat2str(probeTrialsThisBlock));
             end

        % ---------- OPTO selection ----------
        % optoTrialsThisBlock   = [];
        % isTransitionOptoBlock = false;
        if S.GUI.OptoEnabled
            switch S.GUI.OptoSessionType
                case 1 % RandomTrial (pick trials within this block's absolute indices)

                    if currentBlockIndex == 1
                        blockStartTrial = 1;
                    else
                        blockStartTrial = sum(blocks(1:currentBlockIndex - 1)) + 1;
                        fprintf('block Start Trial: %d\n', blockStartTrial);
                    end
                    blockEndTrial = blockStartTrial + blocks(currentBlockIndex) - 1;
                    fprintf('block End Trial: %d\n', blockEndTrial);
                    minTrial = blockStartTrial;
                    maxTrial = blockEndTrial - 2;  % exclude last trial from probes
                    opto_candidate_trials = minTrial:maxTrial;
                    numCandidateTrials = length(opto_candidate_trials);
                    % percentage-based target
                    opto_fraction = S.GUI.OptoFraction;
                    numOpto = round(opto_fraction * numCandidateTrials);
                    % --- Random selection with minimum spacing (bounded) ---
                    Optogap = S.GUI.OptoMinSeparation;   % e.g., 3 -> need >= 3 non-probes between probes
                    Optosep = Optogap + 1;            % index separation
                    Optoselected = [];
                    % keep picking until we hit numProbes (target) or run out of candidates
                    while numel(selected) < numOpto && ~isempty(opto_candidate_trials)
                        % pick one candidate at random
                        o = opto_candidate_trials(randi(numel(opto_candidate_trials)));
                        Optoselected(end+1) = o; %#ok<AGROW>
                        % remove any candidates too close (< sep) to the chosen one
                        opto_candidate_trials = opto_candidate_trials(abs(opto_candidate_trials - o) >= Optosep);
                     end
            
                     optoTrialsThisBlock = sort(Optoselected);
                     optoTrialsByBlock{currentBlockIndex} = optoTrialsThisBlock;
            
                     fprintf('Opto trial indices: %s\n', mat2str(optoTrialsThisBlock));

                case 2 % ProbeTrialOpto (no pre-assignment; handle per‑trial by tying opto to probe)
                    optoTrialsThisBlock = probeTrialsThisBlock; 
                    optoTrialsByBlock{currentBlockIndex} = optoTrialsThisBlock;

                    fprintf('########################################\n');
                    fprintf('[Block %d] ProbeTrialOpto mode: opto will follow probe trials.\n', currentBlockIndex);
                    fprintf('########################################\n');

                case 3  % BlockTransition
                    if currentBlockIndex == 1
                        blockStartTrial = 1+ numWarmupTrials;
                    else
                        blockStartTrial = sum(blocks(1:currentBlockIndex - 1)) + 1;
                        fprintf('block Start Trial: %d\n', blockStartTrial);
                    end

                    % Determine previous block type from the last trial of the previous block
                    if currentBlockIndex > 1
                        prevLastTrial      = blockStartTrial - 1;           % absolute index
                        previousBlockType  = BlockTypes{prevLastTrial};     % you must fill BlockTypes{t} each trial
                    else
                        previousBlockType  = '';
                    end
                    blockStartTrial = sum(blocks(1:currentBlockIndex - 1)) + numWarmupTrials;
                    blockEndTrial = blockStartTrial + blocks(currentBlockIndex) - 1;
                    blockTrialRange = blockStartTrial:blockEndTrial;
                    % If your GUI uses BlockTypeReceiveOpto (1/2/3), alias it here:
                    % if isfield(S.GUI,'BlockTypeReceiveOpto')
                        whichBlockTypeTransition = S.GUI.BlockTypeReceiveOpto;
                        % S.GUI.BlockTypeReceiveOpto;
                    % else
                    %     whichBlockTypeTransition = S.GUI.whichBlockTypeTransition;  % or whatever you named it
                    % end
                
                    % Does this transition qualify for opto under the selected mode?
                    isShortToLong = strcmp(previousBlockType,'short') && strcmp(currentBlockType,'long');
                    isLongToShort = strcmp(previousBlockType,'long')  && strcmp(currentBlockType,'short');
                
                    
                    switch S.GUI.BlockTypeReceiveOpto
                        case 1  % Long Blocks Only (short → long)
                            qualifies = isShortToLong;
				            TransitionCount = 1;
                        case 2  % Short Blocks Only (long → short)
                            qualifies = isLongToShort;
				            TransitionCount = 1;
                        case 3  % Both Blocks (either direction)
                            qualifies = isShortToLong || isLongToShort;
				            % TransitionCount = 0;
                        otherwise
                            qualifies = false;
                    end
                
                    if qualifies
                        % Count *qualifying* transitions only (for alternation)
                        if S.GUI.BlockTypeReceiveOpto == 3 && isLongToShort
                            fprintf('-------------------');
                            fprintf(' isLongToShort and TransitionCount + 1; ');
                            TransitionCount = TransitionCount + 1;
                        end 
                        % Alternation control (0 = all, 1 = alternate)
                        
                        if S.GUI.WhichBlockTransitions == 1
                            allowThisTransition = true;
                        else
                            % Use odd-numbered qualifying transitions; change to ==0 for even if you prefer
                            allowThisTransition = (mod(TransitionCount,2) == 1);
                        end

                
                        if allowThisTransition
                            % Absolute range for *this* block was already computed:
                            %   blockStartTrial, blockEndTrial, blockTrialRange
                            % n = max(0, min(S.GUI.OptoInitialTrials, numel(blockTrialRange)));
                            n = S.GUI.OptoInitialTrials;
                            optoTrialsThisBlock   = blockTrialRange(1:n);
                            
                            
                            optoTrialsByBlock{currentBlockIndex} = optoTrialsThisBlock;
                            
                            % dirStr = ternary(isShortToLong,'short→long','long→short');
                            if isShortToLong
                                dirStr = 'short→long';
                            else
                                dirStr = 'long→short';
                            end
                            fprintf('########################################\n');
                            fprintf('[Block %d] Transition %d %s → OPTO on first %d trials: %s\n', ...
                                currentBlockIndex, TransitionCount, dirStr, n, mat2str(optoTrialsThisBlock));
                            fprintf('########################################\n');
                        else
                            % dirStr = ternary(isShortToLong,'short→long','long→short');
                            if isShortToLong
                                dirStr = 'short→long';
                            else
                                dirStr = 'long→short';
                            end
                            fprintf('########################################\n');
                            fprintf('[Block %d] Transition %d %s → OPTO SKIPPED (alternating).\n', ...
                                currentBlockIndex, TransitionCount, dirStr);
                            fprintf('########################################\n');
                        end
                    else
                        % Not a qualifying transition under the chosen mode
                        fprintf('########################################\n');
                        fprintf('[Block %d] Transition does not match mode (%d). No transition-opto.\n', ...
                            currentBlockIndex, whichBlockTypeTransition);
                        fprintf('########################################\n');
                    end    
            end
        end
    end


        if S.GUI.UseProbeTrials == 1
                  isProbeTrial = ismember(currentTrial, probeTrialsByBlock{currentBlockIndex});
                else
                  isProbeTrial = false;
        end


        if S.GUI.OptoEnabled
            isOptoTrial = ismember(currentTrial, optoTrialsByBlock{currentBlockIndex});
        else
            isOptoTrial = false;

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

        % isOptoTrial = 1;
        % --- Decide opto mode (relative anchor + delay) ---
        OptoTrigThisTrial = isOptoTrial;
        optoAnchor = '';      % 'LED' or 'AIRPUFF'
        optoRelDelay = 0;     % seconds
        % OptoTrigThisTrial = 1;
        if OptoTrigThisTrial
            switch S.GUI.OptoOnset
                case 1   % Same as Airpuff
                    optoAnchor = 'AIRPUFF';
                    optoRelDelay = 0;
                case 2   % 0 ms after LED
                    optoAnchor = 'LED';
                    optoRelDelay = 0;
                case 3   % 200 ms after LED
                    optoAnchor = 'LED';
                    optoRelDelay = 0.2;
                case 4   % 400 ms after LED
                    optoAnchor = 'LED';
                    optoRelDelay = 0.4;
            end
        end
        
        % --- Configure timers (do NOT trigger yet) ---
        %% Global Timers Setup
        % TimerID mapping:
        % 1 → LED (PWM3)
        % 2 → AirPuff (Valve1)
        % 3 → Camera Trigger (BNC1)
        % 4 → Opto (BNC2)

        sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', S.GUI.LED_Dur, 'OnsetDelay', S.GUI.LED_OnsetDelay,...
            'Channel', 'PWM3', 'PulseWidthByte', 255, 'PulseOffByte', 0,...
            'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0); 


        sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', S.GUI.AirPuff_Dur, ...
            'OnsetDelay', AirPuff_OnsetDelay, 'Channel', 'Valve1', ...
            'OnMessage', 1, 'OffMessage', 0);


        sma = SetGlobalTimer(sma, 'TimerID', 3, 'Duration', CamTrigOnDur, 'OnsetDelay', 0, 'Channel', 'BNC1',...
            'OnLevel', 1, 'OffLevel', 0,...
            'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', CamTrigOffDur);

        % % imaging synchronization trial start signal
        % sma = SetGlobalTimer(sma, 'TimerID', 4, 'Duration', 0.068, 'OnsetDelay', 0, 'Channel', 'BNC1',...
        %     'OnLevel', 1, 'OffLevel', 0,...
        %     'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', CamTrigOffDur);        
        
        if OptoTrigThisTrial
            sma = SetGlobalTimer(sma, 'TimerID', 4, ...
                'Duration', S.GUI.OptoDuration, ...
                'OnsetDelay', optoRelDelay, ...   % delay is relative to when we trigger it
                'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0, 'Loop', 0);
        end
        
        % --- Start/ITI etc. (make sure you DO NOT trigger timers here) ---
        sma = AddState(sma, 'Name', 'Start', ...
            'Timer', 0,...
            'StateChangeConditions', {'SoftCode1', 'ITI_Pre'},...
            'OutputActions', {'GlobalTimerTrig', '0100', 'SoftCode', 1, 'RotaryEncoder1', ['Z#' 0]});
            % 'OutputActions', {'GlobalTimerTrig', '0100', 'SoftCode', 1});
        
        sma = AddState(sma, 'Name', 'ITI_Pre', ...
            'Timer', S.GUI.ITI_Pre,...
            'StateChangeConditions', {'Tup', 'CheckEyeOpen'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'CheckEyeOpen', ...
            'Timer', 0,...
            'StateChangeConditions', {'SoftCode1', 'LED_Onset', 'SoftCode2', 'CheckEyeOpenTimeout'},...
            'OutputActions', {'SoftCode', 2});

        % --- LED onset: trigger ONLY LED+Camera here ---
        % Build mask WITHOUT opto
        % (Assuming your mask ordering maps '0001'->Timer1, '0010'->Timer2, '0100'->Timer3, '1000'->Timer4)
        % Use your existing convention: '0011' = LED(1) + Camera(3)

        sma = AddState(sma, 'Name', 'LED_Onset', ...
            'Timer', 0, ...
            'StateChangeConditions', {'GlobalTimer1_Start','MaybeTriggerOpto'}, ...
            'OutputActions', {'GlobalTimerTrig','0011', 'SoftCode',3});
        
        % --- Immediately after LED_Onset, maybe trigger opto (LED-anchored modes) ---
        if OptoTrigThisTrial && strcmp(optoAnchor,'LED')
            sma = AddState(sma, 'Name', 'MaybeTriggerOpto', ...
                'Timer', 0, ...
                'StateChangeConditions', {'Tup','LED_Puff_ISI'}, ...
                'OutputActions', {'GlobalTimerTrig', 4});   % trigger only timer 4
        else
            sma = AddState(sma, 'Name', 'MaybeTriggerOpto', ...
                'Timer', 0, ...
                'StateChangeConditions', {'Tup','LED_Puff_ISI'}, ...
                'OutputActions', {});  % do nothing
        end
        
        % Decide next state after LED based on probe/non-probe
        if isProbeTrial 
            nextStateAfterLED = 'ITI_Post';   % skip AirPuff on probe
        elseif isProbeTrial && OptoTrigThisTrial   
            nextStateAfterLED = 'OptoOnly';
        else
            nextStateAfterLED = 'AirPuff';    % go to AirPuff otherwise
        end

        % --- ISI ---
        sma = AddState(sma, 'Name', 'LED_Puff_ISI', ...
            'Timer', AirPuff_OnsetDelay, ...
            'StateChangeConditions', {'Tup', nextStateAfterLED}, ...
            'OutputActions', {});
        
        % --- AirPuff; if 'Same as Airpuff', trigger opto here with 0 delay ---
        if OptoTrigThisTrial && ~isProbeTrial && strcmp(optoAnchor,'AIRPUFF')
            % Trigger AirPuff (Timer 2) + Opto (Timer 4) together
            sma = AddState(sma, 'Name', 'AirPuff', ...
                'Timer', S.GUI.AirPuff_Dur, ...
                'StateChangeConditions', {'Tup','ITI_Post'}, ...
                'OutputActions', {'SoftCode', 4, 'GlobalTimerTrig', '1010'});
        elseif ~OptoTrigThisTrial && ~isProbeTrial   
                        % Just run AirPuff (Timer 2), no opto trigger here
            sma = AddState(sma, 'Name', 'AirPuff', ...
                'Timer', S.GUI.AirPuff_Dur, ...
                'StateChangeConditions', {'Tup','ITI_Post'}, ...
                'OutputActions', {'SoftCode', 4});

              disp(['@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']);
              disp(['else... AIRPUFF']);
              disp(['@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']);
        % elseif isProbeTrial && OptoTrigThisTrial && strcmp(optoAnchor,'AIRPUFF')
        %      sma = AddState(sma, 'Name', 'OptoOnly', ...
        %         'Timer',0, ...
        %         'StateChangeConditions', {'Tup','ITI_Post'}, ...
        %         'OutputActions', {'GlobalTimerTrig', '1000'}); % your opto line
        % 
        % 
        %       disp(['@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']);
        %       disp(['isProbeTrial && OptoTrigThisTrial && strcmp(optoAnchor)']);
        %       disp(['@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']);
        % 
        % else
        %     % Just run AirPuff (Timer 2), no opto trigger here
        %     sma = AddState(sma, 'Name', 'AirPuff', ...
        %         'Timer', S.GUI.AirPuff_Dur, ...
        %         'StateChangeConditions', {'Tup','ITI_Post'}, ...
        %         'OutputActions', {'SoftCode', 4});
        % 
        %       disp(['@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']);
        %       disp(['else... AIRPUFF']);
        %       disp(['@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']);
        end
        sma = AddState(sma, 'Name', 'OptoOnly', ...
                'Timer',0, ...
                'StateChangeConditions', {'Tup','ITI_Post'}, ...
                'OutputActions', {'GlobalTimerTrig', '1000'}); % your opto line

        % ---------------------------------------------------
        % ITI and clean-up states
        % ---------------------------------------------------
        
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

            BpodSystem.Data.EncoderData{currentTrial} = BpodSystem.PluginObjects.R.readUSBStream(); % Get rotary encoder data captured since last call to R.readUSBStream()
            % Align this trial's rotary encoder timestamps to state machine trial-start (timestamp of '#' command sent from state machine to encoder module in 'TrialStart' state)
            BpodSystem.Data.EncoderData{currentTrial}.Times = BpodSystem.Data.EncoderData{currentTrial}.Times - BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps(1); % Align timestamps to state machine's trial time 0
            BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps = BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps - BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps(1); % Align event timestamps to state machine's trial time 0  
    
            % Update rotary encoder plot
            % might reduce this section to pass
            % BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States, and
            % access them in plot function
            TrialDuration = BpodSystem.Data.TrialEndTimestamp(currentTrial)-BpodSystem.Data.TrialStartTimestamp(currentTrial);
    
            % encoder module doesn't report positions if there is no recent
            % change in position (could probably update this in enc module
            % code, but takes longer)
    
            % impute start and end position and time values for missing data            
            if ~isempty(BpodSystem.Data.EncoderData{currentTrial}.Times) % if some encoder positions reported                
                if currentTrial == 1
                    % if first trial, and if missing position values between start of trial and first
                    % encoder movement, extrapolate from first recorded enc 
                    % position
                    if BpodSystem.Data.EncoderData{currentTrial}.Times(1) > 0
                        BpodSystem.Data.EncoderData{currentTrial}.Times = [0.0 BpodSystem.Data.EncoderData{currentTrial}.Times];
                        BpodSystem.Data.EncoderData{currentTrial}.Positions = [BpodSystem.Data.EncoderData{currentTrial}.Positions(1) BpodSystem.Data.EncoderData{currentTrial}.Positions];
                        BpodSystem.Data.EncoderData{currentTrial}.nPositions = BpodSystem.Data.EncoderData{currentTrial}.nPositions + 1;                
                    end
                else
                    % if > first trial, and if missing position values between start of trial and first
                    % encoder movement, extrapolate from last recorded enc
                    % position of previous trial
                    if BpodSystem.Data.EncoderData{currentTrial}.Times(1) > 0
                        BpodSystem.Data.EncoderData{currentTrial}.Times = [0.0 BpodSystem.Data.EncoderData{currentTrial}.Times];
                        BpodSystem.Data.EncoderData{currentTrial}.Positions = [BpodSystem.Data.EncoderData{currentTrial-1}.Positions(end) BpodSystem.Data.EncoderData{currentTrial}.Positions];
                        BpodSystem.Data.EncoderData{currentTrial}.nPositions = BpodSystem.Data.EncoderData{currentTrial}.nPositions + 1;                
                    end
                end
                % if missing position values after last encoder movement,
                % extrapolate from last recorded enc position
                if BpodSystem.Data.EncoderData{currentTrial}.Times(end) < TrialDuration
                    BpodSystem.Data.EncoderData{currentTrial}.Times = [BpodSystem.Data.EncoderData{currentTrial}.Times TrialDuration];
                    BpodSystem.Data.EncoderData{currentTrial}.Positions = [BpodSystem.Data.EncoderData{currentTrial}.Positions BpodSystem.Data.EncoderData{currentTrial}.Positions(end)];
                    BpodSystem.Data.EncoderData{currentTrial}.nPositions = BpodSystem.Data.EncoderData{currentTrial}.nPositions + 1;                
                end
            else % if no encoder positions reported
                % if first trial, impute position as zero
                if currentTrial == 1
                    BpodSystem.Data.EncoderData{currentTrial}.Times = [0.0 TrialDuration];
                    BpodSystem.Data.EncoderData{currentTrial}.Positions = [0.0 0.0];
                    BpodSystem.Data.EncoderData{currentTrial}.nPositions = 2;
                else
                    % if > first trial, impute positions as extrapolation
                    % from last recorded enc position of previous trial
                    BpodSystem.Data.EncoderData{currentTrial}.Times = [0.0 TrialDuration];
                    BpodSystem.Data.EncoderData{currentTrial}.Positions = [BpodSystem.Data.EncoderData{currentTrial-1}.Positions(end) BpodSystem.Data.EncoderData{currentTrial-1}.Positions(end)];
                    BpodSystem.Data.EncoderData{currentTrial}.nPositions = 2;
                end
                  
            end   
    
            % unwrap circular distance
            wrap = 2880;                     % half encoder range in degrees
            scale = pi / wrap;               % scale units -> radians
            
            BpodSystem.Data.EncoderData{currentTrial}.PositionsUnwrapped = unwrap(BpodSystem.Data.EncoderData{currentTrial}.Positions * scale) / scale;

            %            
            % convert circular to linear distance
            % encoder position is recorded in degrees
            % convert degrees to radians
            % distance traveled is then arc length
            % arc length = wheel_radius * radians
            DegToRad = pi/180;
            BpodSystem.Data.EncoderData{currentTrial}.LinearPositions = S.GUI.WheelRadius * BpodSystem.Data.EncoderData{currentTrial}.PositionsUnwrapped * DegToRad;


            % Encoder data plot
            LastTrialEncoderPlot(BpodSystem.GUIHandles.EncoderAxes, 'update', BpodSystem.Data.EncoderData{currentTrial},...
                TrialDuration);
                % PreVisStimITITimes, ...
                % VisDetect1Times, ...
                % VisualStimulus1Times, ...
                % WaitForPress1Times, ...
                % LeverRetract1Times, ...
                % Reward1Times, ...
                % DidNotPress1Times, ...
                % ITITimes, ...
                % LeverResetPos, ...
                % WaitForPress2Times, ...
                % LeverRetractFinalTimes, ...
                % Reward2Times, ...
                % DidNotPress2Times, ...
                % WaitForPress3Times, ...
                % LeverRetract3Times, ...
                % Reward3Times, ...
                % DidNotPress3Times, ...
                % RewardTimes, ...
                % EarlyPress1Times, ...
                % VisualStimulus2Times, ...
                % PrePress2DelayTimes, ...
                % EarlyPress2Times);        


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
            BpodSystem.Data.RawEvents.Trial{1, currentTrial}.Data.IsOptoTrial = isOptoTrial;

            BpodSystem.Data.ExperimenterInitials = S.GUI.ExperimenterInitials;
            
            % BpodSystem.Data.TrialTypeSequence = S.GUIMeta.TrialTypeSequence.String{S.GUI.TrialTypeSequence};
            % Save both index and label
            BpodSystem.Data.SleepDeprived = S.GUI.SleepDeprived;
            BpodSystem.Data.SleepDeprivedLabel = S.GUIMeta.SleepDeprived.String{S.GUI.SleepDeprived};
            BpodSystem.Data.TrainingStage.String =S.GUIMeta.TrainingStage.String{S.GUI.TrainingStage};
            BpodSystem.Data.Chemogenetics =  S.GUI.ChemogeneticsEnabled;

            % Show the following information to the Experimenter
            protocol_version = 'EBC_Opto_V_5_0';
            % fprintf('*******************************************************\n');

            % fprintf('\bf Bold text\n');  % This will just print "\bf Bold text"
            % Print basic session info

            switch S.GUI.SleepDeprived
                case 1
                    sessionTypeStr = 'Control enabled, No SD';
                case 2
                    sessionTypeStr = 'Pre_EBC SD (SD) enabled, SD was done right before EBC';
                otherwise
                    sessionTypeStr = 'Unknown';
            end

            % fprintf('Session Type: %s\n', sessionTypeStr);
            fprintf('\n================= SESSION SUMMARY =================\n\n');
            fprintf('Protocol Version: %s\n', protocol_version);
            fprintf('Experimenter: %s\n', S.GUI.ExperimenterInitials);
            fprintf('Session Type: %s\n', sessionTypeStr);
            fprintf('warm_up trial counts: %d\n', S.GUI.num_warmup_trials);
            % fprintf('Probe Trials Enabled: %s\n', string(S.GUI.UseProbeTrials));
            % fprintf('Training Stage: %s\n', S.GUIMeta.TrainingStage.String{S.GUI.TrainingStage});

            % Determine and print Training Stage and Block Type info
            if S.GUI.TrainingStage == 1  % Naive → Always Single Block
                trial_type_string = S.GUIMeta.TrialTypeSequence.String{1};  % 'singleBlock'
                fprintf('→ Training Stage: Naive\n');
                fprintf('→ Block Type: %s (Long)\n', trial_type_string);
                fprintf('→ No probe trials\n');
                fprintf('→ Single Block Mode: LED/Puff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_SingleBlock);
            elseif S.GUI.TrainingStage == 2  % Well-Trained → Use GUI selection
                trial_type_string = S.GUIMeta.TrialTypeSequence.String{3}; 
                fprintf('→ Training Stage: Well-Trained\n');
                fprintf('→ Block Type: %s\n', trial_type_string);
                fprintf('→ First %d trials per block are non-probes\n', S.GUI.num_initial_nonprobe_trials_perBlock);
                % Print delay info based on selected TrialTypeSequence
                switch S.GUI.TrialTypeSequence
                    case 1
                        fprintf('→ Single Block Mode (Long): LED/Puff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_SingleBlock);
                    case 2
                        fprintf('→ singleTransition_short_to_long:\n');
                        fprintf('   Short Block AirPuff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Short);
                        fprintf('   Long Block AirPuff Onset Delay  = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Long);
                    case 3
                        fprintf('→ singleTransition_long_to_short:\n');
                        fprintf('   Long Block AirPuff Onset Delay  = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Long);
                        fprintf('   Short Block AirPuff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Short);
                    case 4
                        fprintf('→ doubleBlock_shortFirst:\n');
                        fprintf('   Short Block AirPuff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Short);
                        fprintf('   Long Block AirPuff Onset Delay  = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Long);                
                        
                    case 5
                        fprintf('→ doubleBlock_longFirst:\n');
                        fprintf('   Short Block AirPuff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Short);
                        fprintf('   Long Block AirPuff Onset Delay  = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Long);
                    case 6
                        fprintf('→ doubleBlock_RandomFirst:\n');
                        fprintf('   Short Block AirPuff Onset Delay = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Short);
                        fprintf('   Long Block AirPuff Onset Delay  = %.3f sec\n', S.GUI.AirPuff_OnsetDelay_Long);
                                  
                    % otherwise
                    %     warning('Unknown TrialTypeSequence code: %d\n', S.GUI.TrialTypeSequence);
                end
            else
                fprintf('→ Training Stage: Unknown\n');
                fprintf('→ Block Type: Unknown\n');
            end
            % Trial_level info 
            fprintf('---------------------------------- \n');
            fprintf('Trial_level info \n');
            if isProbeTrial
                fprintf('**********\n');
                fprintf('This is a Probe Trial --> No Airpuff after LED \n');
                fprintf('**********\n');
            else  
                fprintf('This is a Normal Trial, Airpuff after LED \n');
            end

            if isOptoTrial
                fprintf('**********\n');
                fprintf('This is an opto TrialOpto happens at the same time with Airpuff \n');
                fprintf('**********\n');
            else  
                fprintf('This is not an opto Trial, only Airpuff \n');
            end
            fprintf('Current block Type: %s\n', currentBlockType);
            % fprintf('Current Trial Number: %s\n', ExperimenterTrialInfo.TrialNumber);
            fprintf('ISI Current Trial: %.3f sec\n', ISI); % already in seconds
            fprintf('LED Duration Current Trial: %.3f sec\n', S.GUI.LED_Dur);
            fprintf('AirPuff Duration Current Trial: %.3f sec\n', S.GUI.AirPuff_Dur);
            

            SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file
        end

        HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
        if BpodSystem.Status.BeingUsed == 0 % If protocol was stopped, exit the loop

            BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session

            BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
            BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
            BpodSystem.PluginObjects.R = [];             

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

    BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
    BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
    BpodSystem.PluginObjects.R = [];     

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

    try
        BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
        BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
        BpodSystem.PluginObjects.R = [];
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

