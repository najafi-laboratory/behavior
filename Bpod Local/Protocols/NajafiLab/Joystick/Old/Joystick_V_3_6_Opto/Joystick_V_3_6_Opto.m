function Joystick_V_3_6_Opto
try
    % using try/catch for protocol code to allow for disconnecting from
    % modules and logging crash reports when exceptions thrown

    global BpodSystem
    global S
    global M
    
    EnableOpto         = 1;
    updateGUIPos       = 0;
    
    %% init encoder and maestro objects
    BpodSystem.PluginObjects.R = struct;
    M = [];
    
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

    % get computer host name
    % 'COS-3A11406' - Imaging Rig
    % 'COS-3A11427' - Joystick Rig
    % 'COS-3A17904' - Joystick Rig2

    SetRigID(BpodSystem)
    
    %% Assert HiFi module is present + USB-paired (via USB button on console GUI)
    
    disp('Connecting Hifi...');
    BpodSystem.assertModule('HiFi', 1); % The second argument (1) indicates that the HiFi module must be paired with its USB serial port
    
    % Create an instance of the HiFi module
    H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);
    
    %% Connect Maestro
    disp('Connecting Maestro...');     
    switch BpodSystem.Data.RigName
        case 'ImagingRig'
            M = PololuMaestro('COM15'); 
        case 'JoystickRig'
            M = PololuMaestro('COM15'); 
        case 'JoystickRig2'
            M = PololuMaestro('COM8');             
    end 
    
    %% Assert Stepper + Rotary Encoder modules are present + USB-paired (via USB button on console GUI)
    disp('Connecting Encoder...');
    BpodSystem.assertModule('RotaryEncoder', 1); % The second argument [1 1] indicates that both HiFi and RotaryEncoder must be paired with their respective USB serial ports
    BpodSystem.PluginObjects.R = RotaryEncoderModule(BpodSystem.ModuleUSB.RotaryEncoder1); 
    
    %% Define parameters
    [S] = m_InitGUI.SetParams(BpodSystem);
            % assisted trials
        S.GUI.AssistedTrials = 'fnAssistedTrials';
        S.GUIMeta.AssistedTrials.Style = 'pushbutton';
        S.GUI.ATRangeStart = 0;
        S.GUI.ATRangeStop = 0;
        S.GUIPanels.AssistedTrials = {'AssistedTrials', 'ATRangeStart', 'ATRangeStop'};
    
    % move servo out for loading mouse
    M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - 80));
    
    %% Define trials
    BpodSystem.Data.OptoTrialTypes = [];    % store opto trial types as 1-off, 2-on (could later generate as arrays of 0 & 1)
    BpodSystem.Data.OptoTag = [];    % store opto trial types as 1-off, 2-on
    BpodSystem.Data.IsWarmupTrial = [];
    BpodSystem.Data.PressThresholdUsed = [];

    % initial opto trial type generate (random)
    [OptoTrialTypes] = m_Opto.GenOptoTrials(BpodSystem, S);

    BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.
    MaxTrials = 1000;
    TrialTypes = [repmat(1, 1, MaxTrials)]; % default trial type array
    ProbeTrials = [repmat(1, 1, MaxTrials)]; % default probe trials array
    numTrialTypes = 2;
    updateTrialTypeSequence = 1;
    [TrialTypes] =  m_TrialConfig.GenTrials(S, MaxTrials, numTrialTypes, TrialTypes, 1, updateTrialTypeSequence);
    % [EpochIdxs] = m_TrialConfig.GetEpochIdxs;
    
    %% Initialize plots
    
    % Press Outcome Plot
    BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [918 808 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off'); 
    BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
    
    % trial type outcomes for opto
    TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes, OptoTrialTypes, ProbeTrials);
    % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes, OptoTrialTypes, ProbeTrials);
    BpodParameterGUI('init', S); % Initialize parameter GUI plugin
     
    % update gui positions
    set(BpodSystem.ProtocolFigures.ParameterGUI, 'Position', [9 53 1617 866]);

    %% sequence tester - see versions joystick_V_1_3 if trial type testing for proto mods is needed
    % do not remove this section of code, it's very helpful when
    % adding/updating trial type parameters and functions
    inx = 1;
    currentTrial = 1;
    PreviousEnableManualTrialType = S.GUI.EnableManualTrialType;
    PreviousTrialTypeSequence = S.GUI.TrialTypeSequence;
    PreviousNumTrialsPerBlock = S.GUI.NumTrialsPerBlock;
    PreviousBlockLengthMargin = S.GUI.BlockLengthMargin;
    while inx == 0
        inx = input('Set parameters and press enter to continue >', 's');
        if isempty(inx) 
            inx = 0; % refill inx
        end
        % usr_input = ['inx: ', inx];
        % disp(usr_input);
        S = BpodParameterGUI('sync', S);
                
        ManualDeactivated = 0;
        updateTrialTypeSequence = 0;
        if (PreviousEnableManualTrialType ~= S.GUI.EnableManualTrialType) 
            if (PreviousEnableManualTrialType == 1)
                ManualDeactivated = 1;                
            end
            PreviousEnableManualTrialType = S.GUI.EnableManualTrialType;
        end

        if (PreviousTrialTypeSequence ~= S.GUI.TrialTypeSequence) || (ManualDeactivated)
            updateTrialTypeSequence = 1;            
        end
        PreviousTrialTypeSequence = S.GUI.TrialTypeSequence;

        if PreviousNumTrialsPerBlock ~= S.GUI.NumTrialsPerBlock
            updateTrialTypeSequence = 1;
            PreviousNumTrialsPerBlock = S.GUI.NumTrialsPerBlock;
        end

        if PreviousBlockLengthMargin ~= S.GUI.BlockLengthMargin
            updateTrialTypeSequence = 1;
            PreviousBlockLengthMargin = S.GUI.BlockLengthMargin;
        end           
        

        BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.
        % TrialTypes = [];
        

        [TrialTypes] =  m_TrialConfig.GenTrials(S, MaxTrials, numTrialTypes, TrialTypes, currentTrial, updateTrialTypeSequence);
        % m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 0);
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 0);
        [OptoTrialTypes] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, TrialTypes, currentTrial, 1);
        % m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 0);
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 0);
        % [EpochStartIdxs] = m_TrialConfig.GetEpochIdxs;
        % test = abs(diff(TrialTypes))
        % EpochStartIdxs = [1, (find(abs(diff(TrialTypes)) == 1) + 1)]
        ProbeTrialTypes = repmat(0, 1, S.GUI.MaxTrials);
        if (S.GUI.ProbeTrialFraction > 0)
            edgeOffset = 2;
            numTrialsAddedToSequence = 0;
            ProbeTrialTypesAdded = [];
            EpochStartIdxs = [1, (find(abs(diff(TrialTypes)) == 1) + 1), (S.GUI.MaxTrials + 1)];
            % S.GUI.MaxTrials - EpochStartIdxs(end)
            BlockLengths = [diff(EpochStartIdxs) (S.GUI.MaxTrials - EpochStartIdxs(end))];
            BlockEpochIdx = 1;
            while numTrialsAddedToSequence < S.GUI.MaxTrials
                
                numProbeTrialsToAdd = 0;
                ProbeTrialTypesToAdd = [];
                ProbeTrialTypesToAdd = [repmat(0, 1, BlockLengths(BlockEpochIdx) - 2*edgeOffset)];
                numProbeTrialsToAdd = ceil(S.GUI.ProbeTrialFraction * length(ProbeTrialTypesToAdd));
                if BlockLengths(BlockEpochIdx) > 2*edgeOffset
                    numProbeTrialsToAdd_idxs = randperm(length(ProbeTrialTypesToAdd), numProbeTrialsToAdd) + edgeOffset;
                    ProbeTrialTypesToAdd(numProbeTrialsToAdd_idxs) = 1;
                    ProbeTrialTypesToAdd = [ProbeTrialTypesToAdd repmat(0, 1, edgeOffset)];                     
                else                    
                    ProbeTrialTypesToAdd = [ProbeTrialTypesToAdd repmat(0, 1, BlockLengths(BlockEpochIdx))];
                end
                ProbeTrialTypesAdded = [ProbeTrialTypesAdded ProbeTrialTypesToAdd];
                numTrialsAddedToSequence = length(ProbeTrialTypesAdded);
                BlockEpochIdx = BlockEpochIdx + 1;
                % OptoTrialTypes(currentTrial:end) = numProbeTrialsToAdd;
                          
                % if BlockEpochIdx == length(BlockLengths)
                %     disp('debug');
                % end
                % numOpto = S.GUI.EpochTrialStop;
                % if numOpto > BlockLengths(BlockEpochIdx)
                %     numOpto = BlockLengths(BlockEpochIdx);
                % end
                % numNonOpto = BlockLengths(BlockEpochIdx) - S.GUI.EpochTrialStop;
                % if numNonOpto < 1
                %     numNonOpto = 0;
                % end
                % ProbeTrialTypesToAdd = [ProbeTrialTypesToAdd repmat(2, 1, numOpto) repmat(1, 1, numNonOpto)];
                % numTrialsAddedToSequence = numTrialsAddedToSequence + numOpto + numNonOpto;
                % BlockEpochIdx = BlockEpochIdx + 1;
            end
            ProbeTrialTypes(currentTrial:end) = ProbeTrialTypesAdded(1:length(OptoTrialTypes) - currentTrial + 1);

        end
        % m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 0);
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 0);
        currentTrial = currentTrial + 1;
        disp(['currentTrial:', num2str(currentTrial)])
        S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
        % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update',(numTrialTypes+1)-TrialTypes);  
        % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update', TrialTypes);
        
    end


    % init opto trial types
    currentTrial = 1;
    [OptoTrialTypes] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, TrialTypes, currentTrial, 1);    

    % store trial type params
    PreviousEnableManualTrialType = S.GUI.EnableManualTrialType;
    PreviousTrialTypeSequence = S.GUI.TrialTypeSequence;
    PreviousNumTrialsPerBlock = S.GUI.NumTrialsPerBlock;
    PreviousBlockLengthMargin = S.GUI.BlockLengthMargin;
    PreviousSelfTimedMode = S.GUI.SelfTimedMode;

    % keeping for reference, these can be used to change the tick mark
    % label for trial types to improve readability (i.e. short press, long
    % press instead of 1, 2)
    % BpodSystem.GUIHandles.TTOP_Ylabel = 'Press'
    
    %-- Last Trial encoder plot (an online plot included in the protocol folder)
    BpodSystem.ProtocolFigures.EncoderPlotFig = figure('Position', [-2 47 1500 600],'name','Encoder plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
    BpodSystem.GUIHandles.EncoderAxes = axes('Position', [.15 .15 .8 .8]);
    LastTrialEncoderPlot(BpodSystem.GUIHandles.EncoderAxes, 'init', S.GUI.Threshold);
    
    %% state timing plot
    useStateTiming = true;  % Initialize state timing plot
    if ~verLessThan('matlab','9.5') % StateTiming plot requires MATLAB r2018b or newer
        useStateTiming = true;
        StateTiming();
    end
        
    %% Define stimuli and send to analog module
    
    SF = 192000; % Use max supported sampling rate samples/sec, keeping
    %commented for reference
    % SF = 44100; % Use lower sampling rate (samples/sec) to allow for longer duration audio file (max length limited by HiFi)
    H.SamplingRate = SF;
    Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset
    
    IncorrectSound = GenerateWhiteNoise(SF, S.GUI.PunishSoundDuration_s, 1, 1)*S.GUI.IncorrectSoundVolume_percent; % white noise punish sound
    IncorrectSound = ApplySoundEnvelope(IncorrectSound, Envelope);

    EarlyPressPunishSound = GenerateWhiteNoise(SF, S.GUI.EarlyPressPunishSoundDuration_s, 1, 1)*S.GUI.EarlyPressPunishSoundVolume_percent; % white noise punish sound
    EarlyPressPunishSound = ApplySoundEnvelope(EarlyPressPunishSound, Envelope);
    
    % generate audio stim same duration as vis gratings
    AudioStimSound = GenerateSineWave(SF, S.GUI.AudioStimFreq_Hz, S.GUI.GratingDur_s)*S.GUI.AudioStimVolume_percent; % Sampling freq (hz), Sine frequency (hz), duration (s)
    AudioStimSound = ApplySoundEnvelope(AudioStimSound, Envelope);
    
    H.DigitalAttenuation_dB = -35; % Set a comfortable listening level for most headphones (useful during protocol dev).
    H.load(3, IncorrectSound);
    H.load(4, EarlyPressPunishSound);
    
    % Remember values of sound frequencies & durations, so a new one only gets uploaded if it was changed
    LastAudioStimFrequency = S.GUI.AudioStimFreq_Hz;
    LastAudioStimVolume = S.GUI.AudioStimVolume_percent;
    LastPunishSoundDuration = S.GUI.PunishSoundDuration_s;
    LastIncorrectSoundVolume = S.GUI.IncorrectSoundVolume_percent;

    LastEarlyPressPunishSoundDuration = S.GUI.EarlyPressPunishSoundDuration_s;
    LastEarlyPressPunishSoundVolume = S.GUI.EarlyPressPunishSoundVolume_percent;
    
    
    %% Setup video
    
    if isfield(BpodSystem.PluginObjects, 'V') % Clear previous instances of the video server
        BpodSystem.PluginObjects.V = [];
    end

    MonitorID = 1;
    switch BpodSystem.Data.RigName
        case 'ImagingRig'
            MonitorID = 2;
        case 'JoystickRig'
            MonitorID = 1;
        case 'JoystickRig2'
            MonitorID = 1;            
    end    

    BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(MonitorID, 0, [0 0], [180 180], 0); % Assumes second monitor is screen #2. Sync patch = 180x180 pixels
    
    BpodSystem.PluginObjects.V.SyncPatchIntensity = 255; % increased, seems 140 doesn't always trigger BNC high
    
    % Indicate loading
    BpodSystem.PluginObjects.V.loadText(1, 'Loading...', '', 80);
    BpodSystem.PluginObjects.V.play(1);
    
    Ysize = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
    Xsize = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
    
    % compute grating according to square grid of largest pixel dimension
    if Ysize > Xsize
        gratingSize = [Ysize, Ysize]; % Size of grating in pixels
    else
        gratingSize = [Xsize, Xsize]; % Size of grating in pixels
    end
    
    spatialFreq = .005; % Spatial frequency of grating in cycles per pixel % .32
    orientation = 0; % Orientation of grating in degrees
    contrast = 1; % Contrast of grating (0 to 1)
    phase = 0.5;
    
    % Calculate the parameters needed for the grating
    pixPerCycle = 1 / spatialFreq;
    freqPerPixel = 1 / pixPerCycle;
    
    [x, y] = meshgrid(1:gratingSize(1), 1:gratingSize(2));
    
    gray = 0.5 * ones(gratingSize);
    sinGrating = gray + contrast/2 .* sin(freqPerPixel * 2 * pi * (cos(orientation * pi / 180) * x + sin(orientation * pi / 180) * y) + phase);
    sinGrating(sinGrating > 1) = 1; % Cap values above 1 to 1 (white)
    sinGrating(sinGrating < 0) = 0; % Cap values below 0 to 0 (black)
    
    gray = gray(1:Ysize, 1:Xsize); % clip to monitor
    sinGrating = sinGrating(1:Ysize, 1:Xsize); % clip to monitor
    
    % these images are 0 - 1 amplitude, fine for grating_flashes.m file but need to be 0 - 255 for
    % building video this way
    gray = gray * 255;
    sinGrating = sinGrating * 255;
    
    % Query duration of one monitor refresh interval:
    ifi=Screen('GetFlipInterval', BpodSystem.PluginObjects.V.Window); % check via psychtoolbox
    FramesPerSecond = BpodSystem.PluginObjects.V.DetectedFrameRate;
    
    BpodSystem.Data.InterFrameIntervalPsychToolbox = ifi;
    BpodSystem.Data.FramesPerSecondVidPlugin = FramesPerSecond;

    GratingDuration = S.GUI.GratingDur_s; % set duration of grating to stimulus interval
    GrayFixedDuration = S.GUI.ISIOrig_s; % set duration of gray screen to inter stimulus interval
    
    % need an integer number of frames, there is no fractional frame for
    % pscyhtoolbox
    % need an even number of frames for sync patch to alternate light/dark
    % each frame, update later if proto not alternating sync patch each
    % frame
    GratingFrames = convergent(FramesPerSecond * GratingDuration);  % maybe use floor for this? then continue to round up below?
    if (mod(GratingFrames, 2) ~= 0)
        GratingFrames = GratingFrames + 1; % round up to nearest even integer
    end
    GratingDuration = GratingFrames / FramesPerSecond; % convert even rounded number of frames back into duration to calculate video duration
    
    GrayFixedFrames = convergent(FramesPerSecond * GrayFixedDuration);
    if (mod(GrayFixedFrames, 2) ~= 0)
        GrayFixedFrames = GrayFixedFrames + 1; % round up to nearest even integer
    end
    GrayFixedDuration = GrayFixedFrames / FramesPerSecond; % convert even rounded number of frames back into duration to calculate video duration
        
    VideoGrating = repmat(sinGrating, 1, 1, 2); % 2 frames to get sync signal encoded
    VideoGrayFixed = repmat(gray, 1, 1, 2); % 2 frames to get sync signal encoded
    
    BpodSystem.PluginObjects.V.loadVideo(1, VideoGrating);
    BpodSystem.PluginObjects.V.loadVideo(2, VideoGrayFixed);
    
    % compose grating video
    GratingFrame_SyncW = BpodSystem.PluginObjects.V.Videos{1}.Data(1);
    GratingFrame_SyncBlk = BpodSystem.PluginObjects.V.Videos{1}.Data(2);
    GratingBlank = BpodSystem.PluginObjects.V.Videos{1}.Data(3);
    
    GratingPattern = [GratingFrame_SyncW GratingFrame_SyncW];
    
    GratingVideo = [repmat(GratingPattern, 1, GratingFrames/2)];
        
    % compose gray video, fixed ISI
    GrayFrame_SyncW = BpodSystem.PluginObjects.V.Videos{2}.Data(1);
    GrayFrame_SyncBlk = BpodSystem.PluginObjects.V.Videos{2}.Data(2);
    GrayBlank = BpodSystem.PluginObjects.V.Videos{2}.Data(3);
    GrayPattern = [GrayFrame_SyncBlk GrayFrame_SyncBlk];
    GrayProbePattern =  [GrayFrame_SyncW GrayFrame_SyncW];
    GrayVideo = [repmat(GrayPattern, 1, GrayFixedFrames/2)];
    
    % gray probe video same duration as grating video
    ProbeGrayVideo = [repmat(GrayProbePattern, 1, GratingFrames/2)];
    
    % update durations based on number of frames generated
    GratingDur = length(GratingVideo) * (1/FramesPerSecond);
    GrayDur = length(GrayVideo) * (1/FramesPerSecond);
        
    % use init video to set Frame2TTL BNC sync to be low and not miss first frame of vis
    % stim later
    GrayInitBNCSync = [repmat(GrayFrame_SyncW, 1, 120) GrayFrame_SyncBlk];
    BpodSystem.PluginObjects.V.Videos{6} = struct;
    BpodSystem.PluginObjects.V.Videos{6}.nFrames = 121; % + 1 for final frame
    BpodSystem.PluginObjects.V.Videos{6}.Data = GrayInitBNCSync;
       
    LastGratingDuration = S.GUI.GratingDur_s; % Remember value of stim dur so that we only regenerate the grating video if parameter has changed
    LastGrayFixedDuration = S.GUI.ISIOrig_s; % Remember value of pre-perturb gray dur so that we only regenerate the pre-perturb gray video if parameter has changed
    
    BpodSystem.PluginObjects.V.TimerMode = 0;
    pause(1.0); % matlab seems to require a pause here before clearing screen with play(0), 
                % otherwise can get stuck on Psychtoolbox splash screen
                % might need longer delay if purple image hangs on window open
    BpodSystem.PluginObjects.V.play(0);
    BpodSystem.PluginObjects.V.play(6);
    BpodSystem.PluginObjects.V.TimerMode = 2;

    % wait for parameter update and confirm before beginning trial loop
    input('Set parameters and press enter to continue >', 's'); 
    S = BpodParameterGUI('sync', S);

    % update trial types before starting session
    updateTrialTypeSequence = 1;
    [TrialTypes] =  m_TrialConfig.GenTrials(S, MaxTrials, numTrialTypes, TrialTypes, currentTrial, updateTrialTypeSequence);

    % probe trials, consolidate probe trial check/updates later 
    ProbeTrialTypes = repmat(0, 1, S.GUI.MaxTrials);
    if (S.GUI.ProbeTrialFraction > 0)
        edgeOffset = 2;
        numTrialsAddedToSequence = 0;
        ProbeTrialTypesAdded = [];
        EpochStartIdxs = [1, (find(abs(diff(TrialTypes)) == 1) + 1), (S.GUI.MaxTrials + 1)];
        % S.GUI.MaxTrials - EpochStartIdxs(end)
        BlockLengths = [diff(EpochStartIdxs) (S.GUI.MaxTrials - EpochStartIdxs(end))];
        BlockEpochIdx = 1;
        while numTrialsAddedToSequence < S.GUI.MaxTrials
            
            numProbeTrialsToAdd = 0;
            ProbeTrialTypesToAdd = [];
            ProbeTrialTypesToAdd = [repmat(0, 1, BlockLengths(BlockEpochIdx) - 2*edgeOffset)];
            numProbeTrialsToAdd = ceil(S.GUI.ProbeTrialFraction * length(ProbeTrialTypesToAdd));
            % ProbeTrialTypesToAdd = [ProbeTrialTypesToAdd];
            if BlockLengths(BlockEpochIdx) > 2*edgeOffset
                numProbeTrialsToAdd_idxs = randperm(length(ProbeTrialTypesToAdd), numProbeTrialsToAdd) + edgeOffset;
                ProbeTrialTypesToAdd = [ProbeTrialTypesToAdd repmat(0, 1, edgeOffset)];
                ProbeTrialTypesToAdd(numProbeTrialsToAdd_idxs) = 1;
                ProbeTrialTypesToAdd = [ProbeTrialTypesToAdd repmat(0, 1, edgeOffset)];                     
            else                    
                ProbeTrialTypesToAdd = [ProbeTrialTypesToAdd repmat(0, 1, BlockLengths(BlockEpochIdx))];
            end
            ProbeTrialTypesAdded = [ProbeTrialTypesAdded ProbeTrialTypesToAdd];
            length(ProbeTrialTypesToAdd);
            numTrialsAddedToSequence = length(ProbeTrialTypesAdded);
            % numTrialsAddedToSequence
            BlockEpochIdx = BlockEpochIdx + 1;
        end
        ProbeTrialTypes(currentTrial:end) = ProbeTrialTypesAdded(1:length(ProbeTrialTypes) - currentTrial + 1);
    end
    [OptoTrialTypes] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, TrialTypes, currentTrial, 1);
    m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 0);
    % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes, OptoTrialTypes, ProbeTrials);

    % wait for parameter update and confirm before beginning trial loop
    % input('Set parameters and press enter to continue >', 's'); 
    % S = BpodParameterGUI('sync', S);
    
    %% set warmup counter
        
    WarmupTrialsCounter = S.GUI.NumEasyWarmupTrials;
    LastNumEasyWarmupTrials = S.GUI.NumEasyWarmupTrials; % store GUI value to determine if user has changed this param to reset counter
    
    %% Setup rotary encoder module
    
    M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos), 0.5); % move to in position to set joystick zero reference, use half speed
    pause(0.5); % pause for servo to move to in pos
    BpodSystem.PluginObjects.R.setPosition(0); % Set the current position to equal 0
    BpodSystem.PluginObjects.R.thresholds = S.GUI.Threshold;
    BpodSystem.PluginObjects.R.sendThresholdEvents = 'on'; % If on, rotary encoder module will send threshold events to state machine
    BpodSystem.PluginObjects.R.startUSBStream;
    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_Joystick';
    
    %% init any needed experimenter trial info values
    ExperimenterTrialInfo.TrialNumber = 0;
    ExperimenterTrialInfo.VisStimInterruptDetect1Count = 0;
    ExperimenterTrialInfo.VisStimInterruptGray1Count = 0;
    ExperimenterTrialInfo.VisStimInterruptDetect2Count = 0;
    ExperimenterTrialInfo.VisStimInterruptGray2Count = 0;
    ExperimenterTrialInfo.TotalRewardAmount_uL = 0;
    
    % init PreReward delays
    PreRewardDelay_s = 0;
    AutoPreRewardDelay_s = PreRewardDelay_s;
    PreviousEnableAutoPreRewardDelay= 0;  % start zero in case enabled by default, then still get start value for delay

    % init press delays
    PressVisDelay_s = 0;

    % init counters
    TotalRewardAmount_uL = 0;
   
    % check if opto session
    % 1 = 'Opto', 2 = 'Control'
    if S.GUI.SessionType == 2
        m_Opto.EnableOpto = 0;
    else
        m_Opto.EnableOpto = 1;
    end

    % exp notes log
    ExpNotes.numTrials = 0;

    ExpNotes.AssistedTrials = 0;

    if S.GUI.ChemogeneticSession
        ExpNotes.ChemoSession = 'Yes';
        ExpNotes.ChemoDilution = append(num2str(S.GUI.mgCNO), ':', num2str(S.GUI.mlSaline)," mgCNO:mlSaline");
    else
        ExpNotes.ChemoSession = 'No';
    end
   
    if m_Opto.EnableOpto
        switch S.GUI.OptoTrialTypeSeq
            case 1
                ExpNotes.OptoType = 'Random';
                ExpNotes.OnFraction = S.GUI.OnFraction;
            case 2
                ExpNotes.OptoType = 'Random First Block';
            case 3
                ExpNotes.OptoType = 'Off First Block';
            case 4
                ExpNotes.OptoType = 'On First Block';
            case 5
                ExpNotes.OptoType = 'On Epoch';
                ExpNotes.EpochRange = ['1 - ' num2str(S.GUI.EpochTrialStop)];
        end

        if S.GUI.OptoVis1
            ExpNotes.OptoVis1 = 'On';
        else
            ExpNotes.OptoVis1 = 'Off';
        end

        if S.GUI.OptoWaitForPress1
            ExpNotes.OptoWaitForPress1 = 'On';
        else
            ExpNotes.OptoWaitForPress1 = 'Off';
        end

        if S.GUI.OptoVis2
            ExpNotes.OptoVis2 = 'On';
        else
            ExpNotes.OptoVis2 = 'Off';
        end

        if S.GUI.OptoWaitForPress2
            ExpNotes.OptoWaitForPress2 = 'On';
        else
            ExpNotes.OptoWaitForPress2 = 'Off';
        end

        ExpNotes.LEDOnPulseDur_ms = S.GUI.LEDOnPulseDur_ms;
        ExpNotes.LEDOffPulseDur_ms = S.GUI.LEDOffPulseDur_ms;

    end

    ExpNotes.ProbeTrialFraction = S.GUI.ProbeTrialFraction;

    ExpNotes.InitShort = 0;
    ExpNotes.InitLong = 0;

    ExpNotes.FinalShort = 0;
    ExpNotes.FinalLong = 0;

    ExpNotes.PressStep = 0;

    ExpNotes.InitialPreRew = 0;
    ExpNotes.FinalPreRew = 0;

    ExpNotes.RewStep = 0;

    ExpNotes.TotalRewardAmount_uL = 0;

    ExpNotes.ProtoVersion = 'Joystick_V_3_6_Opto';

    %% Main trial loop
    
    for currentTrial = 1:MaxTrials
       
        ExperimenterTrialInfo.TrialNumber = currentTrial;   % check variable states as field/value struct for experimenter info
    
        %% sync trial-specific parameters from GUI

        S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
        S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin    
        
        if (currentTrial == 1)
            ExpNotes.InitShort = S.GUI.PrePress2DelayShort_s;
            ExpNotes.InitLong = S.GUI.PrePress2DelayLong_s;
            ExpNotes.PressStep = S.GUI.AutoDelayStep_s;
            ExpNotes.InitialPreRew = S.GUI.PreRewardDelay_s;
            ExpNotes.RewStep = S.GUI.AutoPreRewardDelayStep_s;
        end

        %% update trial type and sequencing
        % maybe move some of these into GenTrials function
        % update gentrials from opto updates
        ManualDeactivated = 0;
        updateTrialTypeSequence = 0;
        if (PreviousEnableManualTrialType ~= S.GUI.EnableManualTrialType) 
            if (PreviousEnableManualTrialType == 1)
                ManualDeactivated = 1;                
            end
            PreviousEnableManualTrialType = S.GUI.EnableManualTrialType;
        end

        if (PreviousTrialTypeSequence ~= S.GUI.TrialTypeSequence) || (ManualDeactivated)
            updateTrialTypeSequence = 1;            
        end
        PreviousTrialTypeSequence = S.GUI.TrialTypeSequence;

        if PreviousNumTrialsPerBlock ~= S.GUI.NumTrialsPerBlock
            updateTrialTypeSequence = 1;
            PreviousNumTrialsPerBlock = S.GUI.NumTrialsPerBlock;
        end

        if PreviousBlockLengthMargin ~= S.GUI.BlockLengthMargin
            updateTrialTypeSequence = 1;
            PreviousBlockLengthMargin = S.GUI.BlockLengthMargin;
        end        

        if PreviousSelfTimedMode ~= S.GUI.SelfTimedMode
            updateTrialTypeSequence = 1;
            PreviousSelfTimedMode = S.GUI.SelfTimedMode;
        end

        %% update probe trial types warmup
        % no probe trials during warmup
        if WarmupTrialsCounter > 0
            ProbeTrialTypes(currentTrial) = 0;
        end
        BpodSystem.Data.ProbeTrial(currentTrial) = ProbeTrialTypes(currentTrial);

        %% update opto epoch warmup
        if (S.GUI.OptoTrialTypeSeq == 5 && ...
            WarmupTrialsCounter > 0)
            OptoTrialTypes(currentTrial) = 1;
        end        
      
        %% update outcome plot

        % trial type updating needs to be updated with addition of probe
        % trials, epoch opto, block margins, etc before being used during session
        % [TrialTypes] =  m_TrialConfig.GenTrials(S, MaxTrials,
        % numTrialTypes, TrialTypes, currentTrial,
        % updateTrialTypeSequence);   % keep in case we need the ability to
        % change trial types after session start
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 0);


        %% update grating and gray videos
        
        if S.GUI.GratingDur_s ~= LastGratingDuration
            GratingDuration = S.GUI.GratingDur_s; % set duration of grating to stimulus interval        
  
            GratingFrames = convergent(FramesPerSecond * GratingDuration);  % maybe use floor for this? then continue to round up below?
            if (mod(GratingFrames, 2) ~= 0)
                GratingFrames = GratingFrames + 1; % round up to nearest even integer
            end
            
            % compose grating video
            GratingVideo = [repmat(GratingPattern, 1, GratingFrames/2)];
            ProbeGrayVideo = [repmat(GrayProbePattern, 1, GratingFrames/2)];
            
            % update durations based on number of frames generated
            GratingDur = length(GratingVideo) * (1/FramesPerSecond);
            ProbeGrayDur = length(ProbeGrayVideo) * (1/FramesPerSecond);
        end
        if S.GUI.ISIOrig_s ~= LastGrayFixedDuration
            GrayFixedDuration = S.GUI.ISIOrig_s; % set duration of gray screen to inter stimulus interval
    
            GrayFixedFrames = convergent(FramesPerSecond * GrayFixedDuration);
            if (mod(GrayFixedFrames, 2) ~= 0)
                GrayFixedFrames = GrayFixedFrames + 1; % round up to nearest even integer
            end
           
            % compose gray video, fixed ISI
            GrayVideo = [repmat(GrayPattern, 1, GrayFixedFrames/2)];
                   
            % update durations based on number of frames generated
            GrayDur = length(GrayVideo) * (1/FramesPerSecond);        
        end
    
    
        %% update video & audio and change tracking variables for audio and vis stim
     
        % if vis stim dur, audio stim freq, or volume changed then update sound wave
        if (S.GUI.GratingDur_s ~= LastGratingDuration) || ...
            (S.GUI.AudioStimFreq_Hz ~= LastAudioStimFrequency) || ...
            (S.GUI.AudioStimVolume_percent ~= LastAudioStimVolume)
            % generate audio stim with duration of vis stim
            AudioStimSound = GenerateSineWave(SF, S.GUI.AudioStimFreq_Hz, GratingDur)*S.GUI.AudioStimVolume_percent; % Sampling freq (hz), Sine frequency (hz), duration (s)             
            AudioStimSound = ApplySoundEnvelope(AudioStimSound, Envelope);
    
            LastAudioStimFrequency = S.GUI.AudioStimFreq_Hz;
            LastAudioStimVolume = S.GUI.AudioStimVolume_percent;
        end
          
        if (S.GUI.PunishSoundDuration_s ~= LastPunishSoundDuration) || ...
            (S.GUI.IncorrectSoundVolume_percent ~= LastIncorrectSoundVolume)
            IncorrectSound = GenerateWhiteNoise(SF, S.GUI.PunishSoundDuration_s, 1, 2)*S.GUI.IncorrectSoundVolume_percent; % white noise punish sound
            IncorrectSound = ApplySoundEnvelope(IncorrectSound, Envelope);
            H.load(3, IncorrectSound);
            LastPunishSoundDuration = S.GUI.PunishSoundDuration_s;
            LastIncorrectSoundVolume = S.GUI.IncorrectSoundVolume_percent;
        end

        if (S.GUI.EarlyPressPunishSoundDuration_s ~= LastEarlyPressPunishSoundDuration) || ...
            (S.GUI.EarlyPressPunishSoundVolume_percent ~= LastEarlyPressPunishSoundVolume)
            EarlyPressPunishSound = GenerateWhiteNoise(SF, S.GUI.EarlyPressPunishSoundDuration_s, 1, 1)*S.GUI.EarlyPressPunishSoundVolume_percent; % white noise punish sound
            EarlyPressPunishSound = ApplySoundEnvelope(EarlyPressPunishSound, Envelope);
            H.load(4, EarlyPressPunishSound);
            LastEarlyPressPunishSoundDuration = S.GUI.EarlyPressPunishSoundDuration_s;
            LastEarlyPressPunishSoundVolume = S.GUI.EarlyPressPunishSoundVolume_percent;        
        end

        % update for opto interop
        % trial type updating needs to be updated with addition of probe
        % trials, epoch opto, block margins, etc before being used during session
        % [OptoTrialTypes] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, TrialTypes, currentTrial, 0);
        if ~m_Opto.EnableOpto
            OptoStateExpInfo = 'Control';
            OptoTrialExpInfo = 'NA';
        else
            OptoStateExpInfo = 'Opto';
            switch OptoTrialTypes(currentTrial)
                case 1                    
                    OptoTrialExpInfo = 'Opto Off';
                case 2                    
                    OptoTrialExpInfo = 'Opto On';
            end
        end

        % set session data flags to indicate if opto occurs for a given
        % trial. 0 = No Opto,  1 = Opto
        switch OptoTrialTypes(currentTrial)
            case 1
                BpodSystem.Data.OptoTag(currentTrial) = 0;
            case 2
                BpodSystem.Data.OptoTag(currentTrial) = 1;
        end

        % update outcome plot to reflect opto settings
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 0);
                                 
        %% video for opto joystick stim, gray frames for shutter open delay
        VideoOptoDelay = [GrayFrame_SyncW GrayFrame_SyncBlk]; % 60 fps, F1, F2

        VideoOptoDelayDur = (length(VideoOptoDelay) - 1) * (1/FramesPerSecond); % for audio sync, subtract variable start frame 
    
        % if ~ProbeTrialTypes(currentTrial)
        FullOptoVideo = [VideoOptoDelay GratingVideo]; % standard vis stim grating video
        

        % else
        FullOptoProbeVideo = [VideoOptoDelay ProbeGrayVideo]; % gray probe trial video, mimics sync patch of standard video
        % end
         
        FullOptoVideoFrames = length(FullOptoVideo); % DO NOT subtract variable start frame, needed for actual number of frames when using PlayVideo function in VideoPlugin
        VisStim.VisStimDuration = GratingDur;
        ExperimenterTrialInfo.VisStimDuration = VisStim.VisStimDuration;
  
        % load regular video
        BpodSystem.PluginObjects.V.Videos{5} = struct;
        BpodSystem.PluginObjects.V.Videos{5}.nFrames = FullOptoVideoFrames; 
        BpodSystem.PluginObjects.V.Videos{5}.Data = FullOptoVideo;      

        % load gray probe trial video
        BpodSystem.PluginObjects.V.Videos{3} = struct;
        BpodSystem.PluginObjects.V.Videos{3}.nFrames = FullOptoVideoFrames; 
        BpodSystem.PluginObjects.V.Videos{3}.Data = FullOptoProbeVideo;             
              
        % audio for opto delay shift
        OptoAudioStimOffsetNumSamples = VideoOptoDelayDur * SF; % get duration of gray opto delay in number of audio samples for period between audio stim 
    
        OptoAudioStimOffset = zeros(1, OptoAudioStimOffsetNumSamples);

        OptoAudioStimSound = [OptoAudioStimOffset AudioStimSound];

        %% update trial-specific Audio
        % load sound wave to hifi
        % H.load(7, AudioStimSound);
        H.load(7, OptoAudioStimSound);
    
        % toggle punish sound on/off
        Punish_OutputActions = {};
        EarlyPressPunish_OutputActions = {};
        if (S.GUI.IncorrectSound && ...
           ~ProbeTrialTypes(currentTrial))
                Punish_OutputActions = {'HiFi1', ['P' 2]};
                EarlyPressPunish_OutputActions = {'HiFi1', ['P' 3]};          
        end
        
        Punish_OutputActions = [Punish_OutputActions, {'GlobalTimerTrig', '000000010'}];

        %% update trial-specific valve times using calibration table according to set reward amount    
        RewardAmount_uL = 0;
        CenterValveTime = 0;
        if ~ProbeTrialTypes(currentTrial)
            RewardAmount_uL = S.GUI.CenterValveAmount_uL;
            CenterValveTime = GetValveTimes(RewardAmount_uL, [2]); 
        end

        % init reward times, update based on reward rep
        RewardTime = CenterValveTime;

        ExperimenterTrialInfo.CenterValveAmount_uL = S.GUI.CenterValveAmount_uL;
        ExperimenterTrialInfo.CenterValveTime = CenterValveTime;
        
        %% get PreRewardDelay, auto or manual
       
        % when auto pre reward delay becomes enabled, auto pre reward delay starts at
        % gui param
        if S.GUI.EnableAutoPreRewardDelay ~= PreviousEnableAutoPreRewardDelay
            PreviousEnableAutoPreRewardDelay = S.GUI.EnableAutoPreRewardDelay;
            if S.GUI.EnableAutoPreRewardDelay
                disp('set auto pre reward delay to gui param')
                AutoPreRewardDelay_s = S.GUI.PreRewardDelay_s;
            end
        end
        
        % if after first trial, auto pre reward delay enabled, and previous
        % trial was rewarded, and isn't a warmup trial, then increment the auto pre reward delay
        % value
        if  (S.GUI.EnableAutoPreRewardDelay && ...
            (currentTrial > 1) && ...      
            (WarmupTrialsCounter <= 0) && ...
            ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1)))            
            % disp(['WarmupTrialsCounter: ' num2str(WarmupTrialsCounter)])
            
            AutoPreRewardDelay_s = AutoPreRewardDelay_s + S.GUI.AutoPreRewardDelayStep_s;
            disp(['AutoPreRewardDelay_s incremented: ' num2str(AutoPreRewardDelay_s)])
        end
        
        % if auto pre reward delay enabled, then set pre reward delay to
        % the minimum of auto value or upper bound
        % otherwise set to gui param
        if (S.GUI.EnableAutoPreRewardDelay && ...      
            (WarmupTrialsCounter <= 0))
            PreRewardDelay_s = min(AutoPreRewardDelay_s, S.GUI.AutoPreRewardDelayMax_s); 
            S.GUI.PreRewardDelay_s = AutoPreRewardDelay_s;
        else
            PreRewardDelay_s = S.GUI.PreRewardDelay_s;
        end

        disp(['using PreRewardDelay_s: ' num2str(PreRewardDelay_s)]);

        ExperimenterTrialInfo.PreRewardDelay_s = PreRewardDelay_s;

        %% get pre vis stim delay based on trial type gui params, also experimenter info previsdelay/trial type

        switch S.GUI.SelfTimedMode
            case 0 
                ExperimenterTrialInfo.ProtocolMode = 'Visually Guided';
                switch TrialTypes(currentTrial)
                    case 1
                        ExperimenterTrialInfo.TrialType = 'Short Pre Vis Delay';   % check variable states as field/value struct for experimenter info
                    case 2
                        ExperimenterTrialInfo.TrialType = 'Long Pre Vis Delay';   % check variable states as field/value struct for experimenter info
                end
            case 1
                ExperimenterTrialInfo.ProtocolMode = 'Self Timed';
                switch TrialTypes(currentTrial)
                    case 1
                        ExperimenterTrialInfo.TrialType = 'Short Pre Press Delay';   % check variable states as field/value struct for experimenter info
                    case 2
                        ExperimenterTrialInfo.TrialType = 'Long Pre Press Delay';   % check variable states as field/value struct for experimenter info
                end      
        end
       
        % exp info indicate auto delay enabled
        if S.GUI.EnableAutoDelay
            ExperimenterTrialInfo.EnableAutoDelay = 'Auto Delay Enabled';
        end

        % if auto delay enabled, prev trial rewarded, and not warmup, update respective
        % gui param
        if  (currentTrial>1 && ...      
            S.GUI.EnableAutoDelay && ...
            (WarmupTrialsCounter <= 0) && ...
            isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'Reward') && ...
            ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1)))
            switch (TrialTypes(currentTrial-1))
                case 1
                    if (S.GUI.PrePress2DelayLong_s >= 0.100)
                        S.GUI.PrePress2DelayShort_s = min(S.GUI.PrePress2DelayShort_s + S.GUI.AutoDelayStep_s, S.GUI.AutoDelayMaxShort_s);
                        disp(['PrePress2DelayShort_s incremented: ' num2str(S.GUI.PrePress2DelayShort_s)])
                    end
                case 2
                    S.GUI.PrePress2DelayLong_s = min(S.GUI.PrePress2DelayLong_s + S.GUI.AutoDelayStep_s, S.GUI.AutoDelayMaxLong_s);
                    disp(['PrePress2DelayLong_s incremented: ' num2str(S.GUI.PrePress2DelayLong_s)])
            end            
        end

        % use delay for current trial type
        switch TrialTypes(currentTrial)
            case 1
                PressVisDelay_s = S.GUI.PrePress2DelayShort_s;
                disp(['using short delay: ' num2str(PressVisDelay_s)])
            case 2
                PressVisDelay_s = S.GUI.PrePress2DelayLong_s;
                disp(['using long delay: ' num2str(PressVisDelay_s)])
        end                     
                                 
        %% Draw trial-specific Pre-Vis ITI
            
        PreVisStimITI = m_TrialConfig.GetITI(S); % updated V_3_3;
        ExperimenterTrialInfo.PreVisStimITI = PreVisStimITI;

    
        %% fixed dur ITI post for end of trial ITI
            
        EndOfTrialITI = S.GUI.ITI_Post; % updated V_3_3;
        ExperimenterTrialInfo.EndOfTrialITI = EndOfTrialITI;

        %% set state matrix variables        
        
        VisDetectGray1OutputAction = {'RotaryEncoder1', ['E']};
        % different vis stim for probe trials
        if ~ProbeTrialTypes(currentTrial)
            VisDetect2OutputAction = {'SoftCode', 5,'RotaryEncoder1', ['E']};
        else
            VisDetect2OutputAction = {'SoftCode', 3,'RotaryEncoder1', ['E']};
        end
        VisDetectGray2OutputAction = {'RotaryEncoder1', ['E']};
        audStimOpto1 = m_Opto.GetAudStimOpto(S, OptoTrialTypes(currentTrial), 1);
        audStimOpto2 = m_Opto.GetAudStimOpto(S, OptoTrialTypes(currentTrial), 2);

        WaitForPress1_StateChangeConditions = {};        
        WaitForPress1_OutputActions = {'SoftCode', 7,'RotaryEncoder1', ['E']};
        LeverRetract1_OutputActions = {'SoftCode', 8};
        DidNotPress1_OutputActions = {};
        LeverRetract1_StateChangeConditions = {};
    
        VisualStimulus2_OutputActions = [audStimOpto2 'SoftCode', 7,'RotaryEncoder1', ['E']];
        WaitForPress2_StateChangeConditions = {};
        WaitForPress2_OutputActions = {'SoftCode', 7,'RotaryEncoder1', ['E']};       
        DidNotPress2_OutputActions = {};
        LeverRetractFinal_StateChangeConditions = {};
    

        if ProbeTrialTypes(currentTrial)
            Reward_OutputActions = {};
        else
            Reward_OutputActions = {'Valve2', 1};
        end

        PostRewardDelay_StateChangeConditions = {};

        % update after opto proto is defined to create separate function to abstract global timer
        if m_Opto.EnableOpto && (OptoTrialTypes(currentTrial) == 2)
            if S.GUI.OptoVis1
                VisDetectGray1OutputAction = [VisDetectGray1OutputAction , {'GlobalTimerTrig', '000010001'}];
            end
            
            if S.GUI.OptoVis2
                VisDetectGray2OutputAction = [VisDetectGray2OutputAction , {'GlobalTimerTrig', '001000100'}];
            end            

            if S.GUI.OptoVis1 && ~S.GUI.OptoWaitForPress1
                WaitForPress1_OutputActions = [WaitForPress1_OutputActions, {'GlobalTimerCancel', '000010001'}];
            end

            if S.GUI.OptoVis1 && S.GUI.OptoWaitForPress1
                LeverRetract1_OutputActions = [LeverRetract1_OutputActions, {'GlobalTimerCancel', '000010001'}];
                DidNotPress1_OutputActions = [DidNotPress1_OutputActions, {'GlobalTimerCancel', '000010001'}];
            end

            if ~S.GUI.OptoVis1 && S.GUI.OptoWaitForPress1
                LeverRetract1_OutputActions = [LeverRetract1_OutputActions, {'GlobalTimerCancel', '000010001'}];
                DidNotPress1_OutputActions = [DidNotPress1_OutputActions, {'GlobalTimerCancel', '000010001'}];                
            end

            if S.GUI.OptoVis2 && ~S.GUI.OptoWaitForPress2
                WaitForPress2_OutputActions = [WaitForPress2_OutputActions, {'GlobalTimerCancel', '001000100'}];
            end

            if S.GUI.OptoVis2 && S.GUI.OptoWaitForPress2
                DidNotPress2_OutputActions = [DidNotPress2_OutputActions, {'GlobalTimerCancel', '001000100'}];            
            end

            if ~S.GUI.OptoVis2 && S.GUI.OptoWaitForPress2
                DidNotPress2_OutputActions = [DidNotPress2_OutputActions, {'GlobalTimerCancel', '001000100'}];                
            end

            if S.GUI.OptoWaitForPress2
                if S.GUI.SelfTimedMode
                    WaitForPress2_OutputActions = [WaitForPress2_OutputActions, {'GlobalTimerTrig', '001000100'}];
                end 
            end

            LeverRetract1_OutputActions = [LeverRetract1_OutputActions, {'GlobalTimerTrig', '000000010'}];
            Reward_OutputActions = [Reward_OutputActions, {'GlobalTimerTrig', '000000010', 'GlobalTimerCancel', '111111101'}];
        end

        switch S.GUI.Reps
            case 1           
                WaitForPress1_StateChangeConditions = {'Tup', 'DidNotPress1', 'RotaryEncoder1_1', 'Reward'};                
                PostRewardDelay_StateChangeConditions = {'Tup', 'LeverRetract1'};
                LeverRetract1_StateChangeConditions = {'SoftCode1', 'ITI'};                
            case 2
                WaitForPress1_StateChangeConditions = {'Tup', 'DidNotPress1', 'RotaryEncoder1_1', 'LeverRetract1'};

                if ~S.GUI.SelfTimedMode
                    LeverRetract1_StateChangeConditions = {'SoftCode1', 'PreVis2Delay'};
                else
                    LeverRetract1_StateChangeConditions = {'SoftCode1', 'PrePress2Delay'};
                end
                WaitForPress2_StateChangeConditions = {'Tup', 'DidNotPress2', 'RotaryEncoder1_1', 'PreRewardDelay'};
                PostRewardDelay_StateChangeConditions = {'Tup', 'LeverRetractFinal'}; % updated V_3_5
                LeverRetractFinal_StateChangeConditions = {'SoftCode1', 'ITI'};
        end
    
           
            
        %% adjust for warmup trials
        % For warmup trials, wait for press is extended by additional warmup param, after warmup wait for press is S.GUI.PressWindow_s
        
        % check if user has changed number of warmup trials    
        if S.GUI.NumEasyWarmupTrials ~= LastNumEasyWarmupTrials
            WarmupTrialsCounter = S.GUI.NumEasyWarmupTrials;    % update warmup trial counter to current gui param
            LastNumEasyWarmupTrials = S.GUI.NumEasyWarmupTrials;    % store current value to check for change again
        end
        
        % if warmup trial, increase wait for press by gui param PressWindowExtend_s
        Press1Window_s = S.GUI.Press1Window_s;
        Press2Window_s = S.GUI.Press2Window_s;
        if WarmupTrialsCounter > 0
            ExperimenterTrialInfo.Warmup = true;   % check variable states as field/value struct for experimenter info
            ExperimenterTrialInfo.WarmupTrialsRemaining = WarmupTrialsCounter;   % check variable states as field/value struct for experimenter info
            Press1Window_s = S.GUI.Press1Window_s + S.GUI.PressWindowExtend_s;
            Press2Window_s = S.GUI.Press2Window_s + S.GUI.PressWindowExtend_s;

            PressVisDelay_s = min(S.GUI.PrePress2DelayShort_s, PressVisDelay_s);

            Threshold = S.GUI.WarmupThreshold;

            BpodSystem.Data.IsWarmupTrial(currentTrial) = 1;
            

            TrialDifficulty = 1;  % set warmup trial to easy     
        else    
            Threshold = S.GUI.Threshold;

            BpodSystem.Data.IsWarmupTrial(currentTrial) = 0;

            TrialDifficulty = 2;

            ExperimenterTrialInfo.Warmup = false;   % check variable states as field/value struct for experimenter info
            ExperimenterTrialInfo.WarmupTrialsRemaining = 0;   % check variable states as field/value struct for experimenter info
        end

        BpodSystem.Data.PressThresholdUsed(currentTrial) = Threshold;

        % decrement
        if WarmupTrialsCounter > 0
	        WarmupTrialsCounter = WarmupTrialsCounter - 1;
        end
        
        ExperimenterTrialInfo.Press1Window = Press1Window_s;
        ExperimenterTrialInfo.Press2Window = Press2Window_s;

        %% update encoder threshold from params

        BpodSystem.PluginObjects.R.stopUSBStream;   % stop USB streaming to update encoder params
        pause(0.05);
        BpodSystem.PluginObjects.R.thresholds = [Threshold S.GUI.EarlyPressThreshold];    % udate threshold from GUI params
        BpodSystem.PluginObjects.R.startUSBStream;  % restart encoder USB streaming
    
        BpodSystem.Data.TrialData{1, S.GUI.currentTrial}.LeverResetPos = []; % array for lever reset positions
    
        %% difficulty-specific state values
     
        switch TrialDifficulty
            case 1
                ExperimenterTrialInfo.Difficulty = 'EasyWarmup';   % check variable states as field/value struct for experimenter info
            case 2
                ExperimenterTrialInfo.Difficulty = 'Normal';   % check variable states as field/value struct for experimenter info
        end           
                              
        %% add console print for experimenter trial information, these vars are here to make them easier to see when printed on console
        
        ExperimenterTrialInfo.SessionType = OptoStateExpInfo;
        ExperimenterTrialInfo.OptoTrial = OptoTrialExpInfo;  
        ExperimenterTrialInfo.MatlabVer = BpodSystem.Data.MatVer;

        ExperimenterTrialInfo.PrePress2Delay_s = PressVisDelay_s;

        strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
        disp(strExperimenterTrialInfo);          
    
        %% construct state matrix
    
        sma = NewStateMatrix(); % Assemble state matrix
        
        sma = m_Opto.InsertGlobalTimer(BpodSystem, sma, S, VisStim);
      
        sma = AddState(sma, 'Name', 'Start', ...
            'Timer', 0.068,...
            'StateChangeConditions', {'Tup', 'PreVisStimITI', 'RotaryEncoder1_2', 'EarlyPress'},...
            'OutputActions', {['' ...
            'HiFi1'],'*', 'RotaryEncoder1', ['E#' 0], 'BNC1', 1}); % Code to push newly uploaded waves to front (playback) buffers
        
        sma = AddState(sma, 'Name', 'PreVisStimITI', ...
            'Timer', PreVisStimITI,...
            'StateChangeConditions', {'Tup', 'VisDetect1', 'RotaryEncoder1_2', 'EarlyPress'},...
            'OutputActions', {});
        
        %% rep 1
        
        sma = AddState(sma, 'Name', 'VisDetect1', ...
            'Timer', 0.100,...
            'StateChangeConditions', {'Tup', 'VisStimInterruptDetect1', 'BNC1High', 'VisDetectGray1', 'RotaryEncoder1_2', 'EarlyPress1'},...
            'OutputActions', {'SoftCode', 5,'RotaryEncoder1', ['E']});
    
        sma = AddState(sma, 'Name', 'VisDetectGray1', ...
            'Timer', 0.050,...
            'StateChangeConditions', {'Tup', 'VisStimInterruptGray1', 'BNC1High', 'VisualStimulus1', 'RotaryEncoder1_2', 'EarlyPress1'},...
            'OutputActions', VisDetectGray1OutputAction);   % VisDetectGray1OutputAction = {‘RotaryEncoder1’, [‘E’]};     
    
        sma = AddState(sma, 'Name', 'VisualStimulus1', ...
            'Timer', VisStim.VisStimDuration + 0.020,...
            'StateChangeConditions', {'BNC1Low', 'WaitForPress1'},...
            'OutputActions', [audStimOpto1, 'RotaryEncoder1', ['E']]);

        sma = AddState(sma, 'Name', 'WaitForPress1', ...
            'Timer', Press1Window_s,...
            'StateChangeConditions', WaitForPress1_StateChangeConditions,...      % {'Tup', 'DidNotPress1', 'RotaryEncoder1_1', 'LeverRetract1'}
            'OutputActions', WaitForPress1_OutputActions);       % {'SoftCode', 7,'RotaryEncoder1', ['E']}
                   
        sma = AddState(sma, 'Name', 'LeverRetract1', ...
            'Timer', 0,...
            'StateChangeConditions', LeverRetract1_StateChangeConditions,... % When the PC is done resetting the lever, it sends soft code 1 to the state machine
            'OutputActions', LeverRetract1_OutputActions); % On entering the LeverRetract state, send soft code 1 to the PC. The soft code handler will then start resetting the lever.   
       		% Vis-guided: % LeverRetract1_StateChangeConditions = {'SoftCode1', 'PreVis2Delay'};
	 	    % Self-timed: % LeverRetract1_StateChangeConditions = {'SoftCode1', 'PrePress2Delay'};
   		    % LeverRetract1_OutputActions = {'SoftCode', 8};

        sma = AddState(sma, 'Name', 'PreVis2Delay', ...
            'Timer', PressVisDelay_s,...
            'StateChangeConditions', {'RotaryEncoder1_2', 'EarlyPress2', 'Tup', 'VisDetect2'},...
            'OutputActions', {'SoftCode', 7, 'RotaryEncoder1', ['E']}); 

        sma = AddState(sma, 'Name', 'PrePress2Delay', ...
            'Timer', PressVisDelay_s,...
            'StateChangeConditions', {'RotaryEncoder1_2', 'EarlyPress2', 'Tup', 'WaitForPress2'},...
            'OutputActions', {'SoftCode', 7, 'RotaryEncoder1', ['E']});
     
        %% rep 2
    
        sma = AddState(sma, 'Name', 'VisDetect2', ...
            'Timer', 0.100,...
            'StateChangeConditions', {'Tup', 'VisStimInterruptDetect2', 'BNC1High', 'VisDetectGray2', 'RotaryEncoder1_2', 'EarlyPress2'},...
            'OutputActions', VisDetect2OutputAction);  % ~50ms 		% {'SoftCode', 5,'RotaryEncoder1', ['E']}
    
        sma = AddState(sma, 'Name', 'VisDetectGray2', ...
            'Timer', 0.050,...
            'StateChangeConditions', {'Tup', 'VisStimInterruptGray2', 'BNC1High', 'VisualStimulus2', 'RotaryEncoder1_2', 'EarlyPress2'},...
            'OutputActions', VisDetectGray2OutputAction);     % {'RotaryEncoder1', ['E']}     
    
        sma = AddState(sma, 'Name', 'VisualStimulus2', ...
            'Timer', VisStim.VisStimDuration + 0.020,...
            'StateChangeConditions', {'BNC1Low', 'WaitForPress2', 'RotaryEncoder1_1', 'Reward'},...
            'OutputActions', VisualStimulus2_OutputActions);        
   
        sma = AddState(sma, 'Name', 'WaitForPress2', ...
            'Timer', Press2Window_s,...
            'StateChangeConditions', WaitForPress2_StateChangeConditions,...    % {'Tup', 'DidNotPress2', 'RotaryEncoder1_1', 'PreRewardDelay'}
            'OutputActions', WaitForPress2_OutputActions);          % {'SoftCode', 7,'RotaryEncoder1', ['E']};
    
        sma = AddState(sma, 'Name', 'PreRewardDelay', ...
            'Timer' , PreRewardDelay_s, ...
            'StateChangeConditions', {'Tup', 'Reward'}, ...
            'OutputActions', {});        

        sma = AddState(sma, 'Name', 'Reward', ...
            'Timer', RewardTime,...
            'StateChangeConditions', {'Tup', 'PostRewardDelay'},...
            'OutputActions', Reward_OutputActions); 		% {'Valve2', 1}
       
        sma = AddState(sma, 'Name', 'PostRewardDelay', ...
            'Timer', S.GUI.PostRewardDelay_s,...
            'StateChangeConditions', PostRewardDelay_StateChangeConditions,...		% {'Tup', 'LeverRetractFinal'}
            'OutputActions', {});        
        
        sma = AddState(sma, 'Name', 'DidNotPress1', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'Punish'},...
            'OutputActions', DidNotPress1_OutputActions);	% {}
        
        sma = AddState(sma, 'Name', 'DidNotPress2', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'Punish'},...
            'OutputActions', DidNotPress2_OutputActions);	% {}
   
        sma = AddState(sma, 'Name', 'EarlyPress', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'EarlyPressPunish'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'EarlyPress1', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'EarlyPress1Punish'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'EarlyPress2', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'EarlyPress2Punish'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'EarlyPressPunish', ...
            'Timer', S.GUI.EarlyPressPunishSoundDuration_s,...
            'StateChangeConditions', {'Tup', 'Punish_ITI'},...
            'OutputActions', EarlyPressPunish_OutputActions);  

        sma = AddState(sma, 'Name', 'EarlyPress1Punish', ...
            'Timer', S.GUI.EarlyPressPunishSoundDuration_s,...
            'StateChangeConditions', {'Tup', 'Punish_ITI'},...
            'OutputActions', EarlyPressPunish_OutputActions);  		% {'HiFi1', ['P' 3]}

        sma = AddState(sma, 'Name', 'EarlyPress2Punish', ...
            'Timer', S.GUI.EarlyPressPunishSoundDuration_s,...
            'StateChangeConditions', {'Tup', 'Punish_ITI'},...
            'OutputActions', EarlyPressPunish_OutputActions);      	% {'HiFi1', ['P' 3]}   

        sma = AddState(sma, 'Name', 'Punish', ...
            'Timer', S.GUI.PunishSoundDuration_s,...
            'StateChangeConditions', {'Tup', 'Punish_ITI'},...
            'OutputActions', Punish_OutputActions); 
    
        sma = AddState(sma, 'Name', 'VisStimInterruptDetect1', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'LeverRetractFinal'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'VisStimInterruptGray1', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'LeverRetractFinal'},...
            'OutputActions', {});        

        sma = AddState(sma, 'Name', 'VisStimInterruptDetect2', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'LeverRetractFinal'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'VisStimInterruptGray2', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'LeverRetractFinal'},...
            'OutputActions', {});              

        sma = AddState(sma, 'Name', 'Punish_ITI', ...
            'Timer', S.GUI.PunishITI,...
            'StateChangeConditions', {'Tup', 'LeverRetractFinal'},...
            'OutputActions', {}); 

        sma = AddState(sma, 'Name', 'LeverRetractFinal', ...
            'Timer', 0,...
            'StateChangeConditions', LeverRetractFinal_StateChangeConditions,...	% {'SoftCode1', 'ITI'} % Softcode1: Indicate to the state machine that the lever is back in the home position
            'OutputActions', {'SoftCode', 8});

        sma = AddState(sma, 'Name', 'ITI', ...
            'Timer', EndOfTrialITI,...
            'StateChangeConditions', {'Tup', '>exit'},...
            'OutputActions', {'GlobalCounterReset', '111111111'}); 
    
        SendStateMachine(sma); % Send the state matrix to the Bpod device   
        RawEvents = RunStateMachine; % Run the trial and return events
       
        if ~isempty(fieldnames(RawEvents)) % If trial data was returned (i.e. if not final trial, interrupted by user)
            BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
            BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
            BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial); % Adds the trial type of the current trial to data
            BpodSystem.Data.PrePress2Delay(currentTrial) = PressVisDelay_s;
            BpodSystem.Data.Assisted(currentTrial) = 0;
            % m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 1);
            m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 1);
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
    
            PreVisStimITITimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.PreVisStimITI;
            VisDetect1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisDetect1;
            VisualStimulus1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisualStimulus1;
            WaitForPress1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.WaitForPress1;
            LeverRetract1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.LeverRetract1;
            Reward1Times = [NaN NaN]; % removed rew1 and rew2 V_3_3
            DidNotPress1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.DidNotPress1;
             
            ITITimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.ITI;
    
            LeverResetPos = BpodSystem.Data.TrialData{1, currentTrial}.LeverResetPos;
    
            VisualStimulus2Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisualStimulus2;
            WaitForPress2Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.WaitForPress2;
            LeverRetractFinalTimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.LeverRetractFinal;
            Reward2Times = [NaN NaN]; % removed rew1 and rew2 V_3_3
            DidNotPress2Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.DidNotPress2;
    
            % placeholders for press3
            WaitForPress3Times = [0 0];
            LeverRetract3Times = [0 0];
            Reward3Times = [0 0];
            DidNotPress3Times = [0 0];
    
            RewardTimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.Reward; 

            EarlyPress1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.EarlyPress1;  

            PrePress2DelayTimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.PrePress2Delay;

            EarlyPress2Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.EarlyPress2;  
                
            LastTrialEncoderPlot(BpodSystem.GUIHandles.EncoderAxes, 'update', S.GUI.Threshold, BpodSystem.Data.EncoderData{currentTrial},...
                TrialDuration, ...
                PreVisStimITITimes, ...
                VisDetect1Times, ...
                VisualStimulus1Times, ...
                WaitForPress1Times, ...
                LeverRetract1Times, ...
                Reward1Times, ...
                DidNotPress1Times, ...
                ITITimes, ...
                LeverResetPos, ...
                WaitForPress2Times, ...
                LeverRetractFinalTimes, ...
                Reward2Times, ...
                DidNotPress2Times, ...
                WaitForPress3Times, ...
                LeverRetract3Times, ...
                Reward3Times, ...
                DidNotPress3Times, ...
                RewardTimes, ...
                EarlyPress1Times, ...
                VisualStimulus2Times, ...
                PrePress2DelayTimes, ...
                EarlyPress2Times);
  
            SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file  

            if ~isnan(BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisStimInterruptDetect1(1))
                ExperimenterTrialInfo.VisStimInterruptDetect1Count = ExperimenterTrialInfo.VisStimInterruptDetect1Count+1;
            end      

            if ~isnan(BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisStimInterruptGray1(1))
                ExperimenterTrialInfo.VisStimInterruptGray1Count = ExperimenterTrialInfo.VisStimInterruptGray1Count+1;
            end 

            if ~isnan(BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisStimInterruptDetect2(1))
                ExperimenterTrialInfo.VisStimInterruptDetect2Count = ExperimenterTrialInfo.VisStimInterruptDetect2Count+1;
            end      

            if ~isnan(BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisStimInterruptGray2(1))
                ExperimenterTrialInfo.VisStimInterruptGray2Count = ExperimenterTrialInfo.VisStimInterruptGray2Count+1;
            end                

            if ~isnan(BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.Reward(1))
                TotalRewardAmount_uL = TotalRewardAmount_uL + RewardAmount_uL;
                ExperimenterTrialInfo.TotalRewardAmount_uL = TotalRewardAmount_uL;
            end

            BpodSystem.Data.TotalRewardAmount_uL(currentTrial) = TotalRewardAmount_uL; % total rew received up to current trial, session data

        end
        HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
        if BpodSystem.Status.BeingUsed == 0 % If protocol was stopped, exit the loop
            PrintInterruptLog(BpodSystem);

            % exp notes log
            ExpNotes.numTrials = currentTrial;
            ExpNotes.AssistedTrials = num2str(find(BpodSystem.Data.Assisted==1));
            ExpNotes.FinalShort = S.GUI.PrePress2DelayShort_s;
            ExpNotes.FinalLong = S.GUI.PrePress2DelayLong_s;
            ExpNotes.FinalPreRew = S.GUI.PreRewardDelay_s;
            ExpNotes.TotalRewardAmount_uL = TotalRewardAmount_uL;           
            strExpNotes = formattedDisplayText(ExpNotes,'UseTrueFalseForLogical',true, 'NumericFormat','short');
            disp(strExpNotes); 

            BpodSystem.PluginObjects.V = [];
            % BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session
            BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
            BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
            BpodSystem.PluginObjects.R = [];      
            M = [];
            return
        end
    
    end
    
    BpodSystem.PluginObjects.V = [];
    BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
    BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
    BpodSystem.PluginObjects.R = [];
    M = [];

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

    % write the error to file   
    fprintf(fid,'%s\n',MatlabException.identifier);
    fprintf(fid,'%s\n',MatlabException.message);
    fprintf(fid,'%s\n',MatlabException.Correction);

    % print stack
    fprintf(fid, '%s', MatlabException.getReport('extended', 'hyperlinks','off'));

    % close file
    fclose(fid);

    % save workspace variables associated with session to file

    BpodSystem.PluginObjects.V = [];
    disp('Resetting encoder and maestro objects...');
    BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session
    try
        BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
        BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
        BpodSystem.PluginObjects.R = [];
    catch ME2
        disp(ME2.identifier)
        disp(getReport(ME2));
        disp('Encoder not initialized.');
    end
    M = [];
end
end

% generate full envelope for sound given the sound and front part of
% envelope, return enveloped sound
function [SoundWithEnvelope] = ApplySoundEnvelope(Sound, Envelope)
    BackOfTheEnvelope = fliplr(Envelope);   % flipe front envelope to get back envelope
    IdxsBetweenTheEnvelope = length(Sound) - 2 * length(Envelope); % indices between front and back of envelope
    FullEnvelope = [Envelope ones(1, IdxsBetweenTheEnvelope) BackOfTheEnvelope];  % full envelope
    SoundWithEnvelope = Sound .* FullEnvelope;    % apply envelope element-wise
end

% convert maestro motor position from [992 2000] to [-1 1]
function SetMotorPos = ConvertMaestroPos(MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
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

function PrintInterruptLog(BpodSystem)
    % err report log file
    % recording error and stack information to file
    t = datetime;
    session_date = 10000*(year(t)-2000) + 100*month(t) + day(t);
    
    % get session file name
    [SessionFilepath, SessionFileName, Ext] = fileparts(BpodSystem.Path.CurrentDataFile);

    % LogFileDir = 'C:\data analysis\behavior\joystick\logs\';
    LogFileDir = 'C:\Users\gtg424h\OneDrive - Georgia Institute of Technology\Najafi_Lab\0_Data_analysis\InterruptLogs\';    
    % LogFileName = [LogFileDir, num2str(session_date), '_BPod-matlab_interrupt_log_', SessionFileName];    
    LogFileName = [LogFileDir, SessionFileName, '_InterruptCount'];

    % make crash log folder if it doesn't already exist
    [status, msg, msgID] = mkdir(LogFileDir);

    % save workspace variables associated with session
    Data = BpodSystem.Data;
    save(LogFileName, 'Data');
    % add more workspace vars if needed

    %open file
    fid = fopen([LogFileName, '.txt'],'a+');

    %current windows username
    % username=getenv('USERNAME');
    % fprintf(fid,'%s\n\n', username);

    % write session associated with error
    fprintf(fid,'%s\n\n', SessionFileName);

    % date
    fprintf(fid,'%s\n\n', num2str(session_date));

    switch BpodSystem.Data.RigName
        case 'ImagingRig'
            % S.GUI.ServoInPos = 1570.00; % lever start pos
            % S.GUI.ServoOutPos = 34; % can press lever
            fprintf(fid,'%s\n\n', 'Imaging Rig');
        case 'JoystickRig'
            % rig specs
            fprintf(fid,'%s\n\n', 'Joystick Rig - Behavior Room');
        case 'JoystickRig2'
            % rig specs
            fprintf(fid,'%s\n\n', 'Joystick Rig2 - Behavior Room');            
    end

    SessionData = BpodSystem.Data;

    VisStimInterruptDetect1Count = 0;
    VisStimInterruptGray1Count = 0;
    VisStimInterruptDetect2Count = 0;
    VisStimInterruptGray2Count = 0;

    for trial = 1:SessionData.nTrials  

        VisStimInterruptDetect1 = SessionData.RawEvents.Trial{1, trial}.States.VisStimInterruptDetect1;
        if ~isnan(VisStimInterruptDetect1)
            VisStimInterruptDetect1Count = VisStimInterruptDetect1Count + 1;
        end

        VisStimInterruptGray1 = SessionData.RawEvents.Trial{1, trial}.States.VisStimInterruptGray1;
        if ~isnan(VisStimInterruptGray1)
            VisStimInterruptGray1Count = VisStimInterruptGray1Count + 1;
        end
    
        VisStimInterruptDetect2 = SessionData.RawEvents.Trial{1, trial}.States.VisStimInterruptDetect2;
        if ~isnan(VisStimInterruptDetect2)
            VisStimInterruptDetect2Count = VisStimInterruptDetect2Count + 1;
        end

        VisStimInterruptGray2 = SessionData.RawEvents.Trial{1, trial}.States.VisStimInterruptGray2;
        if ~isnan(VisStimInterruptGray2)
            VisStimInterruptGray2Count = VisStimInterruptGray2Count + 1;
        end        


    end

    fprintf(fid,'%s', 'VisStimInterruptDetect1Count    ');
    fprintf(fid,'%s\n', num2str(VisStimInterruptDetect1Count));

    fprintf(fid,'%s', 'VisStimInterruptGray1Count    ');
    fprintf(fid,'%s\n', num2str(VisStimInterruptGray1Count));

    fprintf(fid,'%s', 'VisStimInterruptDetect2Count    ');
    fprintf(fid,'%s\n', num2str(VisStimInterruptDetect2Count));

    fprintf(fid,'%s', 'VisStimInterruptGray2Count    ');
    fprintf(fid,'%s\n', num2str(VisStimInterruptGray2Count)); 
end



% 
% setpref('Internet','E_mail','gtg424h@gatech.edu');
% 
% 
% sendmail('gtg424h@gatech.edu','New subject', ...
% ['Line1 of message' 10 'Line2 of message' 10 ...
% 'Line3 of message' 10 'Line4 of message'])
% 
% 
% setpref('Internet','SMTP_Server','mail.gatech.edu');
% 


