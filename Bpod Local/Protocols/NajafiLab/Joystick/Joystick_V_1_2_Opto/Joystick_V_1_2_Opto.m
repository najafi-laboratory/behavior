function Joystick_V_1_2_Opto
try
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
    
    %% Assert HiFi module is present + USB-paired (via USB button on console GUI)
    
    disp('Connectig encoder...');
    BpodSystem.assertModule('HiFi', 1); % The second argument (1) indicates that the HiFi module must be paired with its USB serial port
    % Create an instance of the HiFi module
    H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1); % The argument is the name of the HiFi module's USB serial port (e.g. COM3)
    
    %% Connect Maestro
    disp('Connectig maestro...');
    M = PololuMaestro('COM15');
    
    
    %% Assert Stepper + Rotary Encoder modules are present + USB-paired (via USB button on console GUI)
    BpodSystem.assertModule('RotaryEncoder', 1); % The second argument [1 1] indicates that both HiFi and RotaryEncoder must be paired with their respective USB serial ports
    BpodSystem.PluginObjects.R = RotaryEncoderModule(BpodSystem.ModuleUSB.RotaryEncoder1); 
    
    
    %% Define parameters
    [S] = m_InitGUI.SetParams(BpodSystem);
    
    % move servo out for loading mouse
    M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - 80));
    
    
    %% Define trials
    BpodSystem.Data.OptoTrialTypes = [];    % store opto trial types as 1-off, 2-on (could later generate as arrays of 0 & 1)
    BpodSystem.Data.OptoTag = [];    % store opto trial types as 1-off, 2-on

    % initial opto trial type generate (random)
    [OptoTrialTypes] = m_Opto.GenOptoTrials(BpodSystem, S);

    BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.
    MaxTrials = 1000;
    TrialTypes = [repmat(1, 1, MaxTrials)]; % default trial type array
    numTrialTypes = 2;
    updateTrialTypeSequence = 1;
    [TrialTypes] =  m_TrialConfig.GenTrials(S, MaxTrials, numTrialTypes, TrialTypes, 1, updateTrialTypeSequence);
    
      
    % % set max number of trials
    % MaxTrials = 1000;
    % numTrialsAddedToSequence = 0;
    
    % % get distribution of trial types
    % numTrialTypes = 2;
    % if S.GUI.ActManualTrialType  
    %     switch S.GUI.ManualTrialType
    %         case 1
    %             TrialTypes = [repmat(1, 1, MaxTrials)];
    %         case 2
    %             TrialTypes = [repmat(2, 1, MaxTrials)];
    %     end
    % else
    %     switch S.GUI.TrialTypeSequence
    %         case 1 % interleaved - random first trial type
    %             % TrialTypes = ceil(rand(1,MaxTrials)*numTrialTypes);
    %             FirstTrialType = ceil(rand(1,1)*numTrialTypes);
    %             switch FirstTrialType
    %                 case 1 % short first
    %                     SecondTrialType = 2;
    %                 case 2 % long first
    %                     SecondTrialType = 1;
    %             end
    %             while numTrialsAddedToSequence < MaxTrials
    %                 TrialTypes = [TrialTypes repmat(FirstTrialType, 1, S.GUI.NumTrialsPerBlock) repmat(SecondTrialType, 1, S.GUI.NumTrialsPerBlock)];
    %                 numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumTrialsPerBlock; 
    %             end
    %             TrialTypes = TrialTypes(1:MaxTrials);
    %         case 2 % interleaved - short first block
    %             while numTrialsAddedToSequence < MaxTrials
    %                 TrialTypes = [TrialTypes repmat(1, 1, S.GUI.NumTrialsPerBlock) repmat(2, 1, S.GUI.NumTrialsPerBlock)];
    %                 numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumTrialsPerBlock; 
    %             end
    %             TrialTypes = TrialTypes(1:MaxTrials);
    %         case 3 % interleaved - long first block
    %             while numTrialsAddedToSequence < MaxTrials
    %                 TrialTypes = [TrialTypes repmat(2, 1, S.GUI.NumTrialsPerBlock) repmat(1, 1, S.GUI.NumTrialsPerBlock)];
    %                 numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumTrialsPerBlock; 
    %             end
    %             TrialTypes = TrialTypes(1:MaxTrials);
    %     end
    % end
         
            
    % initialize anti-bias variables

    %% Initialize plots
    
    % Press Outcome Plot
    % BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
    BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [918 808 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off'); 
    BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
    % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'init',(numTrialTypes+1)-TrialTypes);
    % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'init',TrialTypes);
    TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes, OptoTrialTypes);
    BpodParameterGUI('init', S); % Initialize parameter GUI plugin
    % BpodSystem.ProtocolFigures.ParameterGUI.Scrollable

    % BpodSystem.ProtocolFigures.ParameterGUI.Position = [-1026 51 2324 980];
    % BpodSystem.ProtocolFigures.ParameterGUI.Position = [9 53 1617 708];
    BpodSystem.ProtocolFigures.ParameterGUI.Position = [9 53 1617 782];
     
    %% sequence tester

    inx = 1;
    currentTrial = 1;
    PreviousEnableManualTrialType = S.GUI.EnableManualTrialType;
    PreviousTrialTypeSequence = S.GUI.TrialTypeSequence;
    PreviousNumTrialsPerBlock = S.GUI.NumTrialsPerBlock;
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
        

        BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.
        % TrialTypes = [];
        

        [TrialTypes] =  m_TrialConfig.GenTrials(S, MaxTrials, numTrialTypes, TrialTypes, currentTrial, updateTrialTypeSequence);
        currentTrial = currentTrial + 1;
        disp(['currentTrial:', num2str(currentTrial)])
        S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
        % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update',(numTrialTypes+1)-TrialTypes);  
        % TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update', TrialTypes);
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 0); % this will update the SideOutcomePlot to reflect the current trial type after accounting for any change due to anti-bias control 
    end
    
    [OptoTrialTypes] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, currentTrial, 1);
    m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 0);

    % store trial type params
    PreviousEnableManualTrialType = S.GUI.EnableManualTrialType;
    PreviousTrialTypeSequence = S.GUI.TrialTypeSequence;
    PreviousNumTrialsPerBlock = S.GUI.NumTrialsPerBlock;
    PreviousSelfTimedMode = S.GUI.SelfTimedMode;

    %BpodSystem.GUIHandles.TTOP_Ylabel = 'Press'
    
    %-- Last Trial encoder plot (an online plot included in the protocol folder)
    BpodSystem.ProtocolFigures.EncoderPlotFig = figure('Position', [-2 47 1500 600],'name','Encoder plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
    BpodSystem.GUIHandles.EncoderAxes = axes('Position', [.15 .15 .8 .8]);
    LastTrialEncoderPlot(BpodSystem.GUIHandles.EncoderAxes, 'init', S.GUI.Threshold);

    set(BpodSystem.ProtocolFigures.ParameterGUI, 'Position', [9 53 1617 818]);
    
    %% state timing plot
    useStateTiming = true;  % Initialize state timing plot
    if ~verLessThan('matlab','9.5') % StateTiming plot requires MATLAB r2018b or newer
        useStateTiming = true;
        StateTiming();
    end
    
    
    %% Define stimuli and send to analog module
    
    %SF = 192000; % Use max supported sampling rate samples/sec
    SF = 44100; % Use lower sampling rate (samples/sec) to allow for longer duration audio file (max length limited by HiFi)
    H.SamplingRate = SF;
    Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset
    
    IncorrectSound = GenerateWhiteNoise(SF, S.GUI.PunishSoundDuration_s, 1, 1)*S.GUI.IncorrectSoundVolume_percent; % white noise punish sound
    IncorrectSound = ApplySoundEnvelope(IncorrectSound, Envelope);
    
    % generate audio stim same duration as vis gratings
    AudioStimSound = GenerateSineWave(SF, S.GUI.AudioStimFreq_Hz, S.GUI.GratingDur_s)*S.GUI.AudioStimVolume_percent; % Sampling freq (hz), Sine frequency (hz), duration (s)
    AudioStimSound = ApplySoundEnvelope(AudioStimSound, Envelope);
    
    H.DigitalAttenuation_dB = -35; % Set a comfortable listening level for most headphones (useful during protocol dev).
    H.load(3, IncorrectSound);
    
    % Remember values of sound frequencies & durations, so a new one only gets uploaded if it was changed
    LastAudioStimFrequency = S.GUI.AudioStimFreq_Hz;
    LastAudioStimVolume = S.GUI.AudioStimVolume_percent;
    
    LastPunishSoundDuration = S.GUI.PunishSoundDuration_s;
    
    LastIncorrectSoundVolume = S.GUI.IncorrectSoundVolume_percent;
    
    
    %% Setup video
    
    if isfield(BpodSystem.PluginObjects, 'V') % Clear previous instances of the video server
        BpodSystem.PluginObjects.V = [];
    end
    MonitorID = 1;
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
    
    GratingDuration = S.GUI.GratingDur_s; % set duration of grating to stimulus interval
    GrayFixedDuration = S.GUI.ISIOrig_s; % set duration of gray screen to inter stimulus interval
    
    % need an integer number of frames, there is no fractional frame
    % need an even number of frames for sync patch to alternate
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
    
    BpodSystem.PluginObjects.V.Videos{3} = struct;
    BpodSystem.PluginObjects.V.Videos{3}.nFrames = GratingFrames + 1; % + 1 for final frame
    BpodSystem.PluginObjects.V.Videos{3}.Data = [GratingVideo GratingBlank];
    
    % compose gray video, fixed ISI
    GrayFrame_SyncW = BpodSystem.PluginObjects.V.Videos{2}.Data(1);
    GrayFrame_SyncBlk = BpodSystem.PluginObjects.V.Videos{2}.Data(2);
    GrayBlank = BpodSystem.PluginObjects.V.Videos{2}.Data(3);
    
    GrayPattern = [GrayFrame_SyncBlk GrayFrame_SyncBlk];
    
    GrayVideo = [repmat(GrayPattern, 1, GrayFixedFrames/2)];
    
    % update durations based on number of frames generated
    GratingDur = length(GratingVideo) * (1/FramesPerSecond);
    GrayDur = length(GrayVideo) * (1/FramesPerSecond);
    
    BpodSystem.PluginObjects.V.Videos{4} = struct;
    BpodSystem.PluginObjects.V.Videos{4}.nFrames = GrayFixedFrames + 1; % + 1 for final frame
    BpodSystem.PluginObjects.V.Videos{4}.Data = [GrayVideo GrayBlank];
    
    % try to get Frame2TTL BNC sync to be low and not miss first frame of vis
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


        % update gui positions
    % if updateGUIPos
    %     BpodSystem.ProtocolFigures.ParameterGUI.Position
    % end
    % wait for parameter update and confirm before beginning trial loop
    input('Set parameters and press enter to continue >', 's'); 
    S = BpodParameterGUI('sync', S);
    

    


    %% check difficulty options and ensure correct setting prior to beginning first trial
    
    % define discrete values in distribution
    DifficultyLevels = [1, 2, 3, 4];  % 1 - Easy, 2 - MediumEasy, 3 - MediumHard, 4 - Hard
    
    WarmupTrialsCounter = S.GUI.NumEasyWarmupTrials;
    LastNumEasyWarmupTrials = S.GUI.NumEasyWarmupTrials; % store GUI value to determine if user has changed this param to reset counter
    
    %% Setup rotary encoder module
    
    M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos), 0.5);
    pause(0.5); % pause for servo to move to in pos
    BpodSystem.PluginObjects.R.setPosition(0); % Set the current position to equal 0
    LastKnownEncPos = 0; % last known location of encoder pos
    BpodSystem.PluginObjects.R.thresholds = S.GUI.Threshold;
    BpodSystem.PluginObjects.R.sendThresholdEvents = 'on'; % If on, rotary encoder module will send threshold events to state machine
    BpodSystem.PluginObjects.R.startUSBStream;
    BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_Joystick';
    
    %% init any needed experimenter trial info values
    ExperimenterTrialInfo.TrialNumber = 0;
    ExperimenterTrialInfo.VisStimInterruptCount = 0;
    
    % init auto press delay    
    PressVisDelay_s = 0;
    PrevPressVisDelay_s = PressVisDelay_s;
    % AutoPressVisDelay_s = S.GUI.AutoDelayStart_s;
    AutoPressVisDelay_s = PressVisDelay_s;
    PrevAutoPressVisDelay_s = AutoPressVisDelay_s;
    PreviousEnableAutoDelay = 0;  % start zero in case enabled by default, then still get start value for delay


    % NumDelaySteps = 0;

    %% Main trial loop
    
    for currentTrial = 1:MaxTrials
       
        ExperimenterTrialInfo.TrialNumber = currentTrial;   % check variable states as field/value struct for experimenter info
    
        %% sync trial-specific parameters from GUI
        % S.GUI.NumDelaySteps = NumDelaySteps;
        S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
        S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin    
        
        if S.GUI.SessionType == 2
            m_Opto.EnableOpto = 0;
        else
            m_Opto.EnableOpto = 1;
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

        if PreviousSelfTimedMode ~= S.GUI.SelfTimedMode
            updateTrialTypeSequence = 1;
            PreviousSelfTimedMode = S.GUI.SelfTimedMode;
        end

        [TrialTypes] =  m_TrialConfig.GenTrials(S, MaxTrials, numTrialTypes, TrialTypes, currentTrial, updateTrialTypeSequence);
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 0);


        %% update grating and gray videos
        
        if S.GUI.GratingDur_s ~= LastGratingDuration
            GratingDuration = S.GUI.GratingDur_s; % set duration of grating to stimulus interval        
              
            % need an integer number of frames, there is no fractional frame
            % need an even number of frames for sync patch to alternate
            GratingFrames = convergent(FramesPerSecond * GratingDuration);  % maybe use floor for this? then continue to round up below?
            if (mod(GratingFrames, 2) ~= 0)
                GratingFrames = GratingFrames + 1; % round up to nearest even integer
            end
            % GratingDuration = GratingFrames / FramesPerSecond; % convert even rounded number of frames back into duration to calculate video duration
    
            % compose grating video
            GratingVideo = [repmat(GratingPattern, 1, GratingFrames/2)];
            
            % update durations based on number of frames generated
            GratingDur = length(GratingVideo) * (1/FramesPerSecond);
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
    
    
        %% update video& audio and change tracking variables for audio and vis stim
    
        % if vis stim dur, audio stim freq, or volume changed then update sound wave
        if (S.GUI.GratingDur_s ~= LastGratingDuration) || ...
            (S.GUI.AudioStimFreq_Hz ~= LastAudioStimFrequency) || ...
            (S.GUI.AudioStimVolume_percent ~= LastAudioStimVolume)
            % generate audio stim of 15s (longer than vis stim to go cue would
            % be) then turn sound off at beginning of next state after vis stim
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
            

        % update for opto interop
        [OptoTrialTypes] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, currentTrial, 0);
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

        switch OptoTrialTypes(currentTrial)
            case 1
                BpodSystem.Data.OptoTag(currentTrial) = 0;
            case 2
                BpodSystem.Data.OptoTag(currentTrial) = 1;
        end

        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 0);
    
        %% Construct trial-specific portion of video and add it to base video
                  
        
        %% video for joystick stim
        FullVideo = GratingVideo;
        FullVideoFrames = length(FullVideo);
        VisStim.VisStimDuration = GratingDur;
        ExperimenterTrialInfo.VisStimDuration = VisStim.VisStimDuration;
    
        % load constructed video into the video object
        BpodSystem.PluginObjects.V.Videos{5} = struct;
        BpodSystem.PluginObjects.V.Videos{5}.nFrames = FullVideoFrames; 
        BpodSystem.PluginObjects.V.Videos{5}.Data = FullVideo;
                
        %% update trial-specific Audio
        H.load(7, AudioStimSound);
    
        if S.GUI.IncorrectSound
            Punish_OutputActions = {'HiFi1', ['P' 2]};
        else
            Punish_OutputActions = {};
        end
                
        %% update encoder threshold from params
        % BpodSystem.PluginObjects.R.stopUSBStream;   % stop USB streaming to update encoder params
        % pause(.05);
        % BpodSystem.PluginObjects.R.thresholds = [S.GUI.Threshold S.GUI.EarlyPressThreshold];    % udate threshold from GUI params
        % BpodSystem.PluginObjects.R;
        % BpodSystem.PluginObjects.R.startUSBStream;  % restart encoder USB streaming
        % 
        % BpodSystem.Data.TrialData{1, S.GUI.currentTrial}.LeverResetPos = []; % array for lever reset positions
        
        
        %% resistance level -> actuator
    
    
        %% trial specific audio-vis
        VisDetectOutputAction = {'SoftCode', 5,'RotaryEncoder1', ['E']};       
        audStimOpto = {'HiFi1', ['P', 6]};        % deprecated, use audStimOpto# for separate control of opto segments
        % SCOA.AudStim     = m_Opto.GetAudStimOpto(OptoTrialTypes(currentTrial));

        % reconsolidate how opto outputs are added to each variable based
        % on params later (a bit chaotic from modifying 2AFC code)
        audStimOpto1 = {'HiFi1', ['P', 6]};
        audStimOpto2 = {'HiFi1', ['P', 6]};
        audStimOpto1 = m_Opto.GetAudStimOpto(S, OptoTrialTypes(currentTrial), 1);
        audStimOpto2 = m_Opto.GetAudStimOpto(S, OptoTrialTypes(currentTrial), 2);

        %% state matrix values
       
    
        Reward_OutputActions = {'Valve2', 1, 'SoftCode', 9};
    
        PostRewardDelay_OutputActions = {};
             
        %% update trial-specific valve times according to set reward amount    
        CenterValveTime = GetValveTimes(S.GUI.CenterValveAmount_uL, [2]); 
        CenterValveTimeRep = CenterValveTime * S.GUI.CenterValveAmountRep_percent;
    
        ExperimenterTrialInfo.CenterValveAmount_uL = S.GUI.CenterValveAmount_uL;
        ExperimenterTrialInfo.CenterValveTime = CenterValveTime;
        ExperimenterTrialInfo.CenterValveAmountRep_ul = S.GUI.CenterValveAmount_uL * S.GUI.CenterValveAmountRep_percent;
        ExperimenterTrialInfo.CenterValveTimeRep = CenterValveTimeRep;

        % init reward times, update based on reward rep
        Reward1Time = CenterValveTimeRep;
        Reward2Time = CenterValveTimeRep;       
        Reward3Time = CenterValveTimeRep;

        RewardTime = CenterValveTime;

        %% get pre vis stim delay based on trial type gui params, also experimenter info previsdelay/trial type
        % PressVisDelay_s = -1; % set default, var should only be used when visually guided activated, so err log should indicate if weird state
        switch S.GUI.SelfTimedMode
            case 0 
                ExperimenterTrialInfo.ProtocolMode = 'Visually Guided';
                switch TrialTypes(currentTrial)
                    case 1
                        ExperimenterTrialInfo.TrialType = 'Short Pre Vis Delay';   % check variable states as field/value struct for experimenter info
                    case 2
                        ExperimenterTrialInfo.TrialType = 'Long Pre Vis Delay';   % check variable states as field/value struct for experimenter info
                end
                ExperimenterTrialInfo.PrePress2Delay_s = 'NA';
            case 1
                ExperimenterTrialInfo.ProtocolMode = 'Self Timed';
                ExperimenterTrialInfo.TrialType = 'NA';   % check variable states as field/value struct for experimenter info                
                % ExperimenterTrialInfo.PressVisDelay_s = 'NA';
        end


        % [PressVisDelay_s, ExperimenterTrialInfo]  = m_TrialConfig.GetPressVisDelay(BpodSystem, S, TrialTypes, currentTrial, PressVisDelay_s); % set default, var should only be used when visually guided activated, so err log should indicate if weird state
        % [PressVisDelay_s, ExperimenterTrialInfo]  = m_TrialConfig.GetPressVisDelay(BpodSystem, S, TrialTypes, currentTrial, PressVisDelay_s); % set default, var should only be used when visually guided activated, so err log should indicate if weird state

        % when autodelay becomes enabled, auto delay starts at mode-specific delay
        if S.GUI.EnableAutoDelay ~= PreviousEnableAutoDelay
            PreviousEnableAutoDelay = S.GUI.EnableAutoDelay;
            if S.GUI.EnableAutoDelay
                if S.GUI.SelfTimedMode
                    disp('set auto to self timed start')
                    AutoPressVisDelay_s = S.GUI.PrePress2Delay_s;
                else
                    disp('set auto to vis guided start')
                    AutoPressVisDelay_s = S.GUI.PressVisDelayLong_s;
                end
            end
        end
        if  (currentTrial>1 && ...      
            S.GUI.EnableAutoDelay && ...
            ((TrialTypes(currentTrial) == 2) && ((TrialTypes(currentTrial-1) == 2)) || (S.GUI.SelfTimedMode == 1)) && ...
            isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'Reward') && ...
            ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1)))
            disp(['WarmupTrialsCounter: ' num2str(WarmupTrialsCounter)])
            if WarmupTrialsCounter <= 0
                AutoPressVisDelay_s = AutoPressVisDelay_s + S.GUI.AutoDelayStep_s;
            end
        end
        if S.GUI.EnableAutoDelay
            ExperimenterTrialInfo.EnableAutoDelay = 'Auto Delay Enabled';
        end
        if S.GUI.EnableAutoDelay && (((TrialTypes(currentTrial) == 2)) || (S.GUI.SelfTimedMode == 1))
            % ExperimenterTrialInfo.EnableAutoDelay = 'Auto Delay Enabled';
            if S.GUI.SelfTimedMode == 1          
                PressVisDelay_s = min(AutoPressVisDelay_s, S.GUI.AutoDelayMaxSelf_s);            
                S.GUI.PrePress2Delay_s = PressVisDelay_s;
            else           
                PressVisDelay_s = min(AutoPressVisDelay_s, S.GUI.AutoDelayMaxVis_s);            
            end
            S.GUI.PressVisDelayLong_s = PressVisDelay_s;
            S.GUI.PrePress2Delay_s = PressVisDelay_s;   
        else
            switch S.GUI.SelfTimedMode
                case 0 
                    ExperimenterTrialInfo.ProtocolMode = 'Visually Guided';
                    switch TrialTypes(currentTrial)
                        case 1
                            PressVisDelay_s = S.GUI.PressVisDelayShort_s;
                            % ExperimenterTrialInfo.TrialType = 'Short Pre Vis Delay';   % check variable states as field/value struct for experimenter info
                        case 2
                            PressVisDelay_s = S.GUI.PressVisDelayLong_s;
                            % ExperimenterTrialInfo.TrialType = 'Long Pre Vis Delay';   % check variable states as field/value struct for experimenter info
                    end
                    % ExperimenterTrialInfo.PrePress2Delay_s = 'NA';
                case 1
                    PressVisDelay_s = S.GUI.PrePress2Delay_s;
                    ExperimenterTrialInfo.ProtocolMode = 'Self Timed';
                    ExperimenterTrialInfo.TrialType = 'NA';   % check variable states as field/value struct for experimenter info                
                    % ExperimenterTrialInfo.PressVisDelay_s = 'NA';
                    % ExperimenterTrialInfo.PrePress2Delay_s = S.GUI.PrePress2Delay_s;
            end
        end    
        % if (S.GUI.ResetAutoDelay == 1)
        %     % PressVisDelay_s = S.GUI.AutoDelayStart_s;
        %     % AutoPressVisDelay_s = S.GUI.AutoDelayStart_s;
        %     % S.GUI.PressVisDelayLong_s = PressVisDelay_s;
        %     % S.GUI.PrePress2Delay_s = PressVisDelay_s;            
        %     % S.GUI.NumDelaySteps = 0;
        %     % NumDelaySteps = 0;
        % end
        if PressVisDelay_s ~= PrevPressVisDelay_s
            PrevPressVisDelay_s = PressVisDelay_s;
        end
        if AutoPressVisDelay_s ~= PrevAutoPressVisDelay_s
            PrevAutoPressVisDelay_s = AutoPressVisDelay_s;
        end


        % ExperimenterTrialInfo.PressVisDelay_s = PressVisDelay_s;
        % S.GUI.NumDelaySteps = NumDelaySteps;
        % S = BpodParameterGUI('sync', S);  % Sync parameters with BpodParameterGUI plugin                                                    




        % switch S.GUI.SelfTimedMode
        %     case 0 
        %         ExperimenterTrialInfo.ProtocolMode = 'Visually Guided';
        %         switch TrialTypes(currentTrial)
        %             case 1
        %                 PressVisDelay_s = S.GUI.PressVisDelayShort_s;
        %                 ExperimenterTrialInfo.TrialType = 'Short Pre Vis Delay';   % check variable states as field/value struct for experimenter info
        %             case 2
        %                 PressVisDelay_s = S.GUI.PressVisDelayLong_s;
        %                 ExperimenterTrialInfo.TrialType = 'Long Pre Vis Delay';   % check variable states as field/value struct for experimenter info
        %         end
        %         ExperimenterTrialInfo.PrePress2Delay_s = 'NA';
        %     case 1
        %         PressVisDelay_s = S.GUI.PrePress2Delay_s;
        %         ExperimenterTrialInfo.ProtocolMode = 'Self Timed';
        %         ExperimenterTrialInfo.TrialType = 'NA';   % check variable states as field/value struct for experimenter info                
        %         ExperimenterTrialInfo.PressVisDelay_s = 'NA';
        % %         ExperimenterTrialInfo.PrePress2Delay_s = S.GUI.PrePress2Delay_s;
        % 
        % end     
        % disp(['PressVisDelay_s: ', num2str(PressVisDelay_s)]);
        

        %% update state-flow for number of reps
        % SCOA.StimAct     = m_AVstimConfig.GetStimAct(S, m_Opto.EnableOpto);
        % if m_Opto.EnableOpto
        %         StimAct = [StimAct, {'GlobalTimerCancel', 1}];
        % end

        VisDetect1_StateChangeConditions = {'Tup', 'VisStimInterrupt', 'BNC1High', 'VisualStimulus1', 'RotaryEncoder1_2', 'EarlyPress1'};
        VisDetect2_StateChangeConditions = {'Tup', 'VisStimInterrupt', 'BNC1High', 'VisualStimulus2', 'RotaryEncoder1_2', 'EarlyPress2'};
        VisDetect3_StateChangeConditions = {'Tup', 'VisStimInterrupt', 'BNC1High', 'VisualStimulus3', 'RotaryEncoder1_2', 'EarlyPress'};
        % if PressVisDelay_s == 0
        %     % VisDetect1_StateChangeConditions = {'Tup', 'VisStimInterrupt', 'BNC1High', 'VisualStimulus1'};
        %     VisDetect2_StateChangeConditions = {'Tup', 'VisStimInterrupt', 'BNC1High', 'VisualStimulus2'};
        %     VisDetect3_StateChangeConditions = {'Tup', 'VisStimInterrupt', 'BNC1High', 'VisualStimulus3'};
        % else
            % VisDetect1_StateChangeConditions = {'Tup', 'VisStimInterrupt', 'BNC1High', 'VisualStimulus1', 'RotaryEncoder1_2', 'EarlyPress'};
            
        % end

        % VisDetect2_StateChangeConditions = {'Tup', 'VisStimInterrupt', 'BNC1High', 'VisualStimulus2', 'RotaryEncoder1_2', 'EarlyPress'};
        
        % VisDetect3_StateChangeConditions = {'Tup', 'VisStimInterrupt', 'BNC1High', 'VisualStimulus3', 'RotaryEncoder1_2', 'EarlyPress'};

        LeverRetract1_StateChangeConditions = {};
        LeverRetract2_StateChangeConditions = {};
        LeverRetract3_StateChangeConditions = {};
    
        VisualStimulus1_OutputActions = audStimOpto1;        
        WaitForPress1_StateChangeConditions = {};        
        WaitForPress1_OutputActions = {'SoftCode', 7,'RotaryEncoder1', ['E']};
        % WaitForPress1_OutputActions = {'SoftCode', 7,'RotaryEncoder1', ['E'], 'GlobalTimerCancel', 1};
        LeverRetract1_OutputActions = {'SoftCode', 8};
        Reward1_StateChangeConditions = {'Tup', 'PostReward1Delay'};
        PostReward1Delay_StateChangeConditions = {'Tup', 'LeverRetract1'};
        DidNotPress1_OutputActions = {'SoftCode', 8};
        LeverRetract1_StateChangeConditions = {};
    
        VisualStimulus2_OutputActions = [audStimOpto2 'SoftCode', 7,'RotaryEncoder1', ['E']];
        WaitForPress2_StateChangeConditions = {};
        WaitForPress2_OutputActions = {'SoftCode', 7,'RotaryEncoder1', ['E']};
        % WaitForPress2_OutputActions = {'SoftCode', 7,'RotaryEncoder1', ['E'], 'GlobalTimerCancel', 1};
        Reward2_StateChangeConditions = {};
        PostReward2Delay_StateChangeConditions = {'Tup', 'LeverRetract2'};
        DidNotPress2_OutputActions = {'SoftCode', 8};
        LeverRetract2_StateChangeConditions = {};
    
        WaitForPress3_StateChangeConditions = {};
        Reward3_StateChangeConditions = {};
        PostReward3Delay_StateChangeConditions = {'Tup', 'LeverRetract3'};

        Reward_StateChangeConditions = {'Tup', 'PostRewardDelay'};
        PostRewardDelay_StateChangeConditions = {};

        PreVis2Delay_StateChangeConditions = {};

        PrePress2Delay_StateChangeConditions = {'RotaryEncoder1_2', 'EarlyPress2', 'Tup', 'WaitForPress2'};
    
        % GetAudStimOpto create separate function to abstract global timer
        % trig/cancel out of audStimOpto function (should only be for adding AV stim, do opto separate)
        if m_Opto.EnableOpto && (OptoTrialTypes(currentTrial) == 2)
            if S.GUI.OptoVis1
                WaitForPress1_OutputActions = [WaitForPress1_OutputActions, {'GlobalTimerCancel', 1}];
            end

            if S.GUI.OptoWaitForPress1
                WaitForPress1_OutputActions = [WaitForPress1_OutputActions, {'GlobalTimerTrig', 2}];
                LeverRetract1_OutputActions = [LeverRetract1_OutputActions, {'GlobalTimerCancel', 2}];
                DidNotPress1_OutputActions = [DidNotPress1_OutputActions, {'GlobalTimerCancel', 2}];
                % Reward_OutputActions = [Reward_OutputActions, {'GlobalTimerCancel', 2}];
            end

            if S.GUI.OptoVis2
                WaitForPress2_OutputActions = [WaitForPress2_OutputActions, {'GlobalTimerCancel', 3}];
                % Reward_OutputActions = [Reward_OutputActions, {'GlobalTimerCancel', 3}];
            end

            if S.GUI.OptoWaitForPress2
                WaitForPress2_OutputActions = [WaitForPress2_OutputActions, {'GlobalTimerTrig', 4}];
                DidNotPress2_OutputActions = [DidNotPress2_OutputActions, {'GlobalTimerCancel', 4}];
                % WaitForPress2_OutputActions = [WaitForPress2_OutputActions, {'GlobalTimerCancel', 1}];
            end


            % if S.GUI.OptoWaitForPress1
            %     % WaitForPress1_OutputActions = [WaitForPress1_OutputActions, {'GlobalTimerTrig', 2}];
            % end
            %     % WaitForPress1_OutputActions = [WaitForPress1_OutputActions, 'GlobalTimerTrig', 1];
            %     % WaitForPress2_OutputActions = [WaitForPress2_OutputActions, {'GlobalTimerCancel', 1}];
            % if S.GUI.OptoWaitForPress2
            %     % WaitForPress2_OutputActions = [WaitForPress2_OutputActions, {'GlobalTimerCancel', 1}];
            % end
            %     % Reward_OutputActions = [Reward_OutputActions, {'GlobalTimerCancel', 1}];
                % Reward_OutputActions = [Reward_OutputActions, {'GlobalTimerCancel', 1, 'GlobalTimerCancel', 2, 'GlobalTimerCancel', 3, 'GlobalTimerCancel', 4}];
                % Reward_OutputActions = [Reward_OutputActions, {'GlobalTimer1_End', 'GlobalTimer2_End', 'GlobalTimer3_End', 'GlobalTimer4_End'}];                
                Reward_OutputActions = [Reward_OutputActions, {'GlobalTimerCancel', '1111'}];
        end

        switch S.GUI.Reps
            case 1           
                WaitForPress1_StateChangeConditions = {'Tup', 'DidNotPress1', 'RotaryEncoder1_1', 'Reward'};                
                PostRewardDelay_StateChangeConditions = {'Tup', 'LeverRetract1'};
                LeverRetract1_StateChangeConditions = {'SoftCode1', 'ITI'};                
            case 2
                if S.GUI.Reward_Rep
                    WaitForPress1_StateChangeConditions = {'Tup', 'DidNotPress1', 'RotaryEncoder1_1', 'Reward1'};
                    % Reward1Time = CenterValveTimeRep;
                else
                    WaitForPress1_StateChangeConditions = {'Tup', 'DidNotPress1', 'RotaryEncoder1_1', 'LeverRetract1'};
                end    
                % if ~S.GUI.SelfTimedMode
                %     LeverRetract1_StateChangeConditions = {'SoftCode1', 'PreVis2Delay'};
                % else
                %     LeverRetract1_StateChangeConditions = {'SoftCode1', 'PrePress2Delay'};
                % end
                if ~S.GUI.SelfTimedMode
                    LeverRetract1_StateChangeConditions = {'SoftCode1', 'PreVis2Delay'};
                    if S.GUI.VisStim2Enable
                        % LeverRetract1_StateChangeConditions = {'SoftCode1', 'VisDetect2'};
                        % if PressVisDelay_s == 0
                        %     PreVis2Delay_StateChangeConditions = {'Tup', 'VisDetect2'};
                        % else
                        PreVis2Delay_StateChangeConditions = {'RotaryEncoder1_2', 'EarlyPress2', 'Tup', 'VisDetect2'};
                        % end                    
                    else
                        % LeverRetract1_StateChangeConditions = {'SoftCode1', 'WaitForPress2'};
                        % if PressVisDelay_s == 0
                        %     PreVis2Delay_StateChangeConditions = {'Tup', 'WaitForPress2'};
                        % else
                        PreVis2Delay_StateChangeConditions = {'RotaryEncoder1_2', 'EarlyPress2', 'Tup', 'WaitForPress2'};
                        % end                                        
                    end
                else
                    LeverRetract1_StateChangeConditions = {'SoftCode1', 'PrePress2Delay'};
                end
                WaitForPress2_StateChangeConditions = {'Tup', 'DidNotPress2', 'RotaryEncoder1_1', 'Reward'};
                % Reward2_StateChangeConditions = {'Tup', 'PostReward2Delay'};
                PostRewardDelay_StateChangeConditions = {'Tup', 'LeverRetract2'};
                LeverRetract2_StateChangeConditions = {'SoftCode1', 'ITI'};
            case 3
                if S.GUI.Reward_Rep
                    WaitForPress1_StateChangeConditions = {'Tup', 'DidNotPress1', 'RotaryEncoder1_1', 'Reward1'};
                    % Reward1Time = CenterValveTimeRep;
                    WaitForPress2_StateChangeConditions = {'Tup', 'DidNotPress2', 'RotaryEncoder1_1', 'Reward2'};
                    % Reward2Time = CenterValveTimeRep;
                else
                    WaitForPress1_StateChangeConditions = {'Tup', 'DidNotPress1', 'RotaryEncoder1_1', 'LeverRetract1'};
                    WaitForPress2_StateChangeConditions = {'Tup', 'DidNotPress2', 'RotaryEncoder1_1', 'LeverRetract2'};
                end            
                % if S.GUI.VisStim2Enable
                %     % LeverRetract1_StateChangeConditions = {'SoftCode1', 'VisDetect2'};
                % else
                %     % LeverRetract1_StateChangeConditions = {'SoftCode1', 'WaitForPress2'};
                % end
                Reward2_StateChangeConditions = {'Tup', 'PostReward2Delay'};
                LeverRetract2_StateChangeConditions = {'SoftCode1', 'VisDetect3'};
                WaitForPress3_StateChangeConditions = {'Tup', 'DidNotPress3', 'RotaryEncoder1_1', 'Reward'};
                Reward3_StateChangeConditions = {'Tup', 'PostReward3Delay'};
                PostRewardDelay_StateChangeConditions = {'Tup', 'LeverRetract3'};
                LeverRetract3_StateChangeConditions = {'SoftCode1', 'ITI'};
        end
    
        %% training level specific state matrix values
    
        AuToPress_StateChangeConditions = {};
    
        switch S.GUI.TrainingLevel
            case 1 % Habituation        
                VisualStimulus1_StateChangeConditions = {'Tup', 'AuToPress'}; % not really important, have as much as you want, eat at Mho's
                AuToPress_StateChangeConditions = {'SoftCode2', 'Reward', 'SoftCode1', 'DidNotPress1'};
                ExperimenterTrialInfo.TrainingLevel = 'Habituation';
            case 2 % naive        
                VisualStimulus1_StateChangeConditions = {'Tup', 'WaitForPress1'};
                ExperimenterTrialInfo.TrainingLevel = 'Naive';
            case 3 % Mid 1 Trained
                VisualStimulus1_StateChangeConditions = {'Tup', 'WaitForPress1'};
                ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 1';
            case 4 % Mid 2 Trained
                VisualStimulus1_StateChangeConditions = {'Tup', 'WaitForPress1'};
                ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 2';
            case 5 % well trained
                VisualStimulus1_StateChangeConditions = {'Tup', 'WaitForPress1'};
                ExperimenterTrialInfo.TrainingLevel = 'Well Trained';
        end
    
        %% Draw trial-specific and difficulty-defined TimeOutPunish from exponential distribution
    

    
    
        %% Draw trial-specific ITI from exponential distribution
    
        [ITI] = m_TrialConfig.GetITI(S);
    
        ExperimenterTrialInfo.ITI = ITI;

        [TimeOutPunish] = m_TrialConfig.GetTimeOutPunish(S);
    
        ExperimenterTrialInfo.TimeOutPunish = TimeOutPunish;
                                      
        %% Split ITI into Pre-Vis Stim duration and end of trial duration.    
        PreVisStimITI = 0.200;
        ExperimenterTrialInfo.PreVisStimITI = PreVisStimITI;
        if ITI-PreVisStimITI >= 0
            EndOfTrialITI = ITI-PreVisStimITI;
        else
            EndOfTrialITI = 0;
        end
        ExperimenterTrialInfo.EndOfTrialITI = EndOfTrialITI; 
    
        %% Draw trial-specific and difficulty-defined TimeOutPunish from exponential distribution
            
        EndOfTrialITI = EndOfTrialITI + TimeOutPunish;
        ExperimenterTrialInfo.EndOfTrialITI = EndOfTrialITI;
            
        %% adjust for warmup trials
        % For warmup trials, wait for press is extended by additional warmup param, after warmup wait for press is S.GUI.PressWindow_s
        
        % check if user has changed number of warmup trials    
        if S.GUI.NumEasyWarmupTrials ~= LastNumEasyWarmupTrials
            WarmupTrialsCounter = S.GUI.NumEasyWarmupTrials;    % update warmap trial counter to current gui param
            LastNumEasyWarmupTrials = S.GUI.NumEasyWarmupTrials;    % store currenct value to check for change again
        end
        
        % if warmup trial, increase wait for press by gui param PressWindowExtend_s
        PressWindow_s = S.GUI.PressWindow_s;
        if WarmupTrialsCounter > 0
            ExperimenterTrialInfo.Warmup = true;   % check variable states as field/value struct for experimenter info
            ExperimenterTrialInfo.WarmupTrialsRemaining = WarmupTrialsCounter;   % check variable states as field/value struct for experimenter info
            PressWindow_s = S.GUI.PressWindow_s + S.GUI.PressWindowExtend_s;
            
            % half press delays for warmup
            if S.GUI.SelfTimedMode
                PressVisDelay_s = PressVisDelay_s /2;
            else
                PressVisDelay_s = min(S.GUI.PressVisDelayShort_s, PressVisDelay_s);
            end



            Threshold = S.GUI.WarmupThreshold;

            TrialDifficulty = 1;  % set warmup trial to easy     
        else    
            Threshold = S.GUI.Threshold;



            TrialDifficulty = 2;
            ExperimenterTrialInfo.Warmup = false;   % check variable states as field/value struct for experimenter info
            ExperimenterTrialInfo.WarmupTrialsRemaining = 0;   % check variable states as field/value struct for experimenter info
        end   

        %% update encoder threshold from params

        

        BpodSystem.PluginObjects.R.stopUSBStream;   % stop USB streaming to update encoder params
        pause(.05);
        % BpodSystem.PluginObjects.R.thresholds = [S.GUI.Threshold S.GUI.EarlyPressThreshold];    % udate threshold from GUI params
        BpodSystem.PluginObjects.R.thresholds = [Threshold S.GUI.EarlyPressThreshold];    % udate threshold from GUI params
        BpodSystem.PluginObjects.R;
        BpodSystem.PluginObjects.R.startUSBStream;  % restart encoder USB streaming
    
        BpodSystem.Data.TrialData{1, S.GUI.currentTrial}.LeverResetPos = []; % array for lever reset positions
    
        ExperimenterTrialInfo.PressWindow = PressWindow_s;
    
        % decrement
        if WarmupTrialsCounter > 0
	        WarmupTrialsCounter = WarmupTrialsCounter - 1;
        end
    
    
        %% difficulty-specific state values
        % switch TrialDifficulty
        %     case 1
        %         ExperimenterTrialInfo.Difficulty = 'Easy';   % check variable states as field/value struct for experimenter info
        %     case 2
        %         ExperimenterTrialInfo.Difficulty = 'Medium-Easy';   % check variable states as field/value struct for experimenter info
        %     case 3
        %         ExperimenterTrialInfo.Difficulty = 'Medium-Hard';   % check variable states as field/value struct for experimenter info
        %     case 4
        %         ExperimenterTrialInfo.Difficulty = 'Hard';   % check variable states as field/value struct for experimenter info
        % end       
        switch TrialDifficulty
            case 1
                ExperimenterTrialInfo.Difficulty = 'EasyWarmup';   % check variable states as field/value struct for experimenter info
            case 2
                ExperimenterTrialInfo.Difficulty = 'Normal';   % check variable states as field/value struct for experimenter info
        end           
                          
        %% After warmup trials, reset wait_dur: with every non early-choice trial, increase it by wait_dur_step
    
        %[wait_dur] = m_TrialConfig.GetWaitDur(BpodSystem, S, wait_dur, currentTrial, LastNumEasyWarmupTrials, VisStim.VisStimDuration);
        wait_dur = 0;
    
    
        %ExperimenterTrialInfo.WaitDuration = wait_dur;
    
    
        %% add console print for experimenter trial information
        ExperimenterTrialInfo.SessionType = OptoStateExpInfo;
        ExperimenterTrialInfo.OptoTrial = OptoTrialExpInfo;  
        ExperimenterTrialInfo.MatlabVer = BpodSystem.Data.MatVer;

        ExperimenterTrialInfo.PressVisDelay_s = PressVisDelay_s;

        strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
        disp(strExperimenterTrialInfo);          
    
        %% construct state matrix
    
        sma = NewStateMatrix(); % Assemble state matrix
        sma = m_Opto.InsertGlobalTimer(sma, S, VisStim);
        % needs more conditions here when adding difficulty levels
        % sma = SetCondition(sma, 1, CorrectPort, 1); % Condition 1: Correct Port is high (licking)
        % sma = SetCondition(sma, 2, IncorrectPort, 1); % Condition 2: Incorrect Port is high (licking)
    
        % testing behavior port4in port4out for use as sync signal from camera
        sma = SetCondition(sma, 3, 'Port4', 1); % Condition 3: Sync signal high
        sma = SetCondition(sma, 4, 'Port4', 0); % Condition 4: Sync signal low
        sma = SetCondition(sma, 5, 'Port2', 1); % Condition 5: center port is high (licking)
    
        CounterNumber = 1;
        TargetEventName = 'Port3In';
        Threshold = 3;

        sma = SetGlobalCounter(sma, CounterNumber, TargetEventName, Threshold);

        sma = AddState(sma, 'Name', 'Start', ...
            'Timer', 0.068,...
            'StateChangeConditions', {'Tup', 'PreVisStimITI'},...
            'OutputActions', {'HiFi1','*', 'RotaryEncoder1', ['E#' 0], 'BNC1', 1, 'PWM3', 255}); % Code to push newly uploaded waves to front (playback) buffers
        
        sma = AddState(sma, 'Name', 'PreVisStimITI', ...
            'Timer', PreVisStimITI,...
            'StateChangeConditions', {'Tup', 'VisDetect1'},...
            'OutputActions', {});
    
    %% rep 1
        
        sma = AddState(sma, 'Name', 'VisDetect1', ...
            'Timer', 0.100,...
            'StateChangeConditions', VisDetect1_StateChangeConditions,...
            'OutputActions', VisDetectOutputAction);
    
        sma = AddState(sma, 'Name', 'VisualStimulus1', ...
            'Timer', VisStim.VisStimDuration,...
            'StateChangeConditions', VisualStimulus1_StateChangeConditions,...
            'OutputActions', VisualStimulus1_OutputActions);
    
        sma = AddState(sma, 'Name', 'AuToPress', ...
            'Timer', 0,...
            'StateChangeConditions', AuToPress_StateChangeConditions,...
            'OutputActions', {'RotaryEncoder1', ['E'], 'SoftCode', 9});
    
        sma = AddState(sma, 'Name', 'WaitForPress1', ...
            'Timer', PressWindow_s,...
            'StateChangeConditions', WaitForPress1_StateChangeConditions,...
            'OutputActions', WaitForPress1_OutputActions);
            % 'OutputActions', {'SoftCode', 7,'RotaryEncoder1', ['E']});
    
        % sma = AddState(sma, 'Name', 'Reward1', ...
        %     'Timer', Reward1Time,...
        %     'StateChangeConditions', Reward1_StateChangeConditions,...
        %     'OutputActions', Reward_OutputActions);
    
        sma = AddState(sma, 'Name', 'PostReward1Delay', ...
            'Timer', S.GUI.PostRewardDelay_s,...
            'StateChangeConditions', PostReward1Delay_StateChangeConditions,...
            'OutputActions', PostRewardDelay_OutputActions);    
                   
        sma = AddState(sma, 'Name', 'LeverRetract1', ...
            'Timer', 0,...
            'StateChangeConditions', LeverRetract1_StateChangeConditions,... % When the PC is done resetting the lever, it sends soft code 1 to the state machine
            'OutputActions', LeverRetract1_OutputActions); % On entering the LeverRetract state, send soft code 1 to the PC. The soft code handler will then start resetting the lever.   
            % 'OutputActions', {'SoftCode', 8}); % On entering the LeverRetract state, send soft code 1 to the PC. The soft code handler will then start resetting the lever.   
       
        sma = AddState(sma, 'Name', 'PreVis2Delay', ...
            'Timer', PressVisDelay_s,...
            'StateChangeConditions', PreVis2Delay_StateChangeConditions,...
            'OutputActions', {'SoftCode', 7, 'RotaryEncoder1', ['E']}); 

        sma = AddState(sma, 'Name', 'PrePress2Delay', ...
            'Timer', PressVisDelay_s,...
            'StateChangeConditions', PrePress2Delay_StateChangeConditions,...
            'OutputActions', {'SoftCode', 7, 'RotaryEncoder1', ['E']});


        % sma = AddState(sma, 'Name', 'PrePress2Delay', ...
        %     'Timer', S.GUI.PrePress2Delay_s,...
        %     'StateChangeConditions', PrePress2Delay_StateChangeConditions,...
        %     'OutputActions', {'SoftCode', 7, 'RotaryEncoder1', ['E']});        
    %% rep 2
    
        sma = AddState(sma, 'Name', 'VisDetect2', ...
            'Timer', 0.100,...
            'StateChangeConditions', VisDetect2_StateChangeConditions,...
            'OutputActions', VisDetectOutputAction);  % ~50ms
    
        sma = AddState(sma, 'Name', 'VisualStimulus2', ...
            'Timer', VisStim.VisStimDuration,...
            'StateChangeConditions', {'Tup', 'WaitForPress2', 'RotaryEncoder1_1', 'Reward'},...
            'OutputActions', VisualStimulus2_OutputActions);
   
        sma = AddState(sma, 'Name', 'WaitForPress2', ...
            'Timer', PressWindow_s,...
            'StateChangeConditions', WaitForPress2_StateChangeConditions,...
            'OutputActions', WaitForPress2_OutputActions);  
            % 'OutputActions', {'SoftCode', 7,'RotaryEncoder1', ['E']});  
    
        sma = AddState(sma, 'Name', 'Reward2', ...
            'Timer', Reward2Time,...
            'StateChangeConditions', Reward2_StateChangeConditions,...
            'OutputActions', Reward_OutputActions);
    
        sma = AddState(sma, 'Name', 'PostReward2Delay', ...
            'Timer', S.GUI.PostRewardDelay_s,...
            'StateChangeConditions', PostReward2Delay_StateChangeConditions,...
            'OutputActions', PostRewardDelay_OutputActions);      
    
        sma = AddState(sma, 'Name', 'LeverRetract2', ...
            'Timer', 0,...
            'StateChangeConditions', LeverRetract2_StateChangeConditions,...
            'OutputActions', {'SoftCode', 8});
       
    %% rep 3
        
        sma = AddState(sma, 'Name', 'VisDetect3', ...
            'Timer', 0.100,...
            'StateChangeConditions', VisDetect3_StateChangeConditions,...
            'OutputActions', VisDetectOutputAction);
    
        sma = AddState(sma, 'Name', 'VisualStimulus3', ...
            'Timer', VisStim.VisStimDuration,...
            'StateChangeConditions', {'Tup', 'WaitForPress3'},...
            'OutputActions', audStimOpto);
    
        sma = AddState(sma, 'Name', 'WaitForPress3', ...
            'Timer', PressWindow_s,...
            'StateChangeConditions', WaitForPress3_StateChangeConditions,...
            'OutputActions', {'SoftCode', 7, 'RotaryEncoder1', 'E'}); % Port1 LED turns on to indicate that the lever is ready to press
                
        sma = AddState(sma, 'Name', 'Reward3', ...
            'Timer', Reward3Time,...
            'StateChangeConditions', Reward3_StateChangeConditions,...
            'OutputActions', Reward_OutputActions);
    
        sma = AddState(sma, 'Name', 'PostReward3Delay', ...
            'Timer', S.GUI.PostRewardDelay_s,...
            'StateChangeConditions', PostReward3Delay_StateChangeConditions,...
            'OutputActions', PostRewardDelay_OutputActions);  
    
        sma = AddState(sma, 'Name', 'LeverRetract3', ...
            'Timer', 0,...
            'StateChangeConditions', LeverRetract3_StateChangeConditions,...
            'OutputActions', {'SoftCode', 8});    
    
    %% reps complete
        % sma = AddState(sma, 'Name', 'Reward', ...
        %     'Timer', CenterValveTime,...
        %     'StateChangeConditions', CenterReward_StateChangeConditions,...
        %     'OutputActions', CenterReward_OutputActions); 
    
        sma = AddState(sma, 'Name', 'Reward1', ...
            'Timer', Reward1Time,...
            'StateChangeConditions', Reward1_StateChangeConditions,...
            'OutputActions', Reward_OutputActions);

        sma = AddState(sma, 'Name', 'Reward', ...
            'Timer', RewardTime,...
            'StateChangeConditions', Reward_StateChangeConditions,...
            'OutputActions', Reward_OutputActions);

        sma = AddState(sma, 'Name', 'PostRewardDelay', ...
            'Timer', S.GUI.PostRewardDelay_s,...
            'StateChangeConditions', PostRewardDelay_StateChangeConditions,...
            'OutputActions', PostRewardDelay_OutputActions);        
        
        sma = AddState(sma, 'Name', 'DidNotPress1', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'Punish'},...
            'OutputActions', DidNotPress1_OutputActions);
        
        sma = AddState(sma, 'Name', 'DidNotPress2', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'Punish'},...
            'OutputActions', DidNotPress2_OutputActions);
    
        sma = AddState(sma, 'Name', 'DidNotPress3', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'Punish'},...
            'OutputActions', {'SoftCode', 8});    
    
        %%
        sma = AddState(sma, 'Name', 'EarlyPress', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'Punish'},...
            'OutputActions', {'SoftCode', 8});
        %%
        sma = AddState(sma, 'Name', 'EarlyPress1', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'Punish'},...
            'OutputActions', {'SoftCode', 8});

        sma = AddState(sma, 'Name', 'EarlyPress2', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', 'Punish'},...
            'OutputActions', {'SoftCode', 8});

        sma = AddState(sma, 'Name', 'Punish', ...
            'Timer', S.GUI.PunishSoundDuration_s,...
            'StateChangeConditions', {'Tup', 'ITI'},...
            'OutputActions', Punish_OutputActions); 
    
        sma = AddState(sma, 'Name', 'VisStimInterrupt', ...
            'Timer', 0,...
            'StateChangeConditions', {'Tup', '>exit'},...
            'OutputActions', {});
        
        sma = AddState(sma, 'Name', 'ITI', ...
            'Timer', EndOfTrialITI,...
            'StateChangeConditions', {'Tup', '>exit'},...
            'OutputActions', {'GlobalCounterReset', 1}); 
    
        SendStateMachine(sma); % Send the state matrix to the Bpod device        
        RawEvents = RunStateMachine; % Run the trial and return events
       
        if ~isempty(fieldnames(RawEvents)) % If trial data was returned (i.e. if not final trial, interrupted by user)
            BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
            BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
            BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial); % Adds the trial type of the current trial to data
            m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, 1);
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
            Reward1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.Reward1;
            DidNotPress1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.DidNotPress1;
             
            ITITimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.ITI;
    
            LeverResetPos = BpodSystem.Data.TrialData{1, currentTrial}.LeverResetPos;
    
            VisualStimulus2Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisualStimulus2;
            WaitForPress2Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.WaitForPress2;
            LeverRetract2Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.LeverRetract2;
            Reward2Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.Reward2;
            DidNotPress2Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.DidNotPress2;
    
            WaitForPress3Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.WaitForPress3;
            LeverRetract3Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.LeverRetract3;
            Reward3Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.Reward3;
            DidNotPress3Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.DidNotPress3;
    
            RewardTimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.Reward;

            % UPDATE
            % EarlyPressTimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.EarlyPress;  

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
                LeverRetract2Times, ...
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



                % EarlyPressTimes, ...
                % VisualStimulus2Times, ...
                % PrePress2DelayTimes);
    
            SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file
    
            if ~isnan(BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisStimInterrupt(1))
                ExperimenterTrialInfo.VisStimInterruptCount = ExperimenterTrialInfo.VisStimInterruptCount+1;
            end
        end
        HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
        if BpodSystem.Status.BeingUsed == 0 % If protocol was stopped, exit the loop
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
    BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session
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

function SetMotorPos = ConvertMaestroPos(MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end

