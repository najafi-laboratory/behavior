function joystick_double_motor_timing_202601

% using try/catch for protocol code to allow for disconnecting from
% modules and logging crash reports when exceptions thrown

global BpodSystem
global S
global M

EnableOpto         = 1;

%% init encoder and maestro objects
BpodSystem.PluginObjects.R = struct;
M = [];

%% Import scriptsBpod

m_Plotter = Plotter;
m_InitGUI = InitGUI;
m_TrialConfig = TrialConfig;
m_AVstim = AVstimConfig;
m_Opto = OptoConfig(EnableOpto);

%% Turn off Bpod LEDs

% This code will disable the state machine status LED
BpodSystem.setStatusLED(0);

BpodSystem.Data.MatVer = version;

% counter for plot filenames
BpodSystem.Data.PlotCntr = 1;
% current trial var for bpod global (needed for pdf plots)
BpodSystem.Data.CurrentTrial = 1;

BpodSystem = SetRigID(BpodSystem);

%% Assert HiFi module is present + USB-paired (via USB button on console GUI)

switch BpodSystem.Data.RigName
    case {'JoystickRig3', 'JoystickRig4'}
        % Program sound server
        if ~isfield(BpodSystem.PluginObjects, 'Sound')
            BpodSystem.PluginObjects.Sound = PsychToolboxAudio;
        end
    otherwise
        disp('Connecting Hifi...');
        BpodSystem.assertModule('HiFi', 1); % The second argument (1) indicates that the HiFi module must be paired with its USB serial port
        
        % Create an instance of the HiFi module
        H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);
end



%% Connect Maestro
disp('Connecting Maestro...');     
switch BpodSystem.Data.RigName
    case 'ImagingRig'
        M = PololuMaestro('COM15'); 
    case 'JoystickRig1'
        M = PololuMaestro('COM15'); 
    case 'JoystickRig2'
        M = PololuMaestro('COM8'); 
    case 'JoystickRig3'
        M = PololuMaestro('COM10'); 
    case 'JoystickRig4'
        M = PololuMaestro('COM8');
end 

%% Assert Stepper + Rotary Encoder modules are present + USB-paired (via USB button on console GUI)
disp('Connecting Encoder...');
BpodSystem.assertModule('RotaryEncoder', 1); % The second argument [1 1] indicates that both HiFi and RotaryEncoder must be paired with their respective USB serial ports
BpodSystem.PluginObjects.R = RotaryEncoderModule(BpodSystem.ModuleUSB.RotaryEncoder1); 

%% Define parameters
[S] = m_InitGUI.SetParams(BpodSystem);

% move servo out for loading mouse
M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - 80));

%% Define trials    
BpodSystem.Data.OptoTrialTypes = [];    % store opto trial types as 1-off, 2-on (could later generate as arrays of 0 & 1)
BpodSystem.Data.AssistTrial = [];
BpodSystem.Data.OptoTag = [];    % store opto trial types as 1-off, 2-on
BpodSystem.Data.IsWarmupTrial = [];
BpodSystem.Data.PressThresholdUsed = [];

% initial opto trial type generate (random)
[OptoTrialTypes, BpodSystem] = m_Opto.GenOptoTrials(BpodSystem, S);

BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.
MaxTrials = 1000;
TrialTypes = ones(1, MaxTrials);
ProbeTrialTypes = zeros(1, MaxTrials);
numTrialTypes = 2;
updateTrialTypeSequence = 1;
[TrialTypes] =  m_TrialConfig.GenTrials(S, MaxTrials, numTrialTypes, TrialTypes, 1, updateTrialTypeSequence);

%% Initialize plots

% Press Outcome Plot
BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [918 808 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off'); 
BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);

% trial type outcomes for opto
TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes, OptoTrialTypes, ProbeTrialTypes);
BpodParameterGUI('init', S); % Initialize parameter GUI plugin
 
% update gui positions
set(BpodSystem.ProtocolFigures.ParameterGUI, 'Position', [9 50 1863 948]);

% init opto trial types
currentTrial = 1;
[OptoTrialTypes, BpodSystem] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, TrialTypes, currentTrial, 1);    

% store trial type params
PreviousEnableManualTrialType = S.GUI.EnableManualTrialType;
PreviousTrialTypeSequence = S.GUI.TrialTypeSequence;
PreviousNumTrialsPerBlock = S.GUI.NumTrialsPerBlock;
PreviousBlockLengthMargin = S.GUI.BlockLengthMargin;

%-- Last Trial encoder plot (an online plot included in the protocol folder)
BpodSystem.ProtocolFigures.EncoderPlotFig = figure('Position', [-2 47 900 300],'name','Encoder plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
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

switch BpodSystem.Data.RigName
    case {'JoystickRig3', 'JoystickRig4'}
        disp('Setup Sound Card Audio...');

    otherwise
        disp('Setup Hifi Audio...');
        H.SamplingRate = SF;
end

Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset

IncorrectSound = GenerateWhiteNoise(SF, S.GUI.PunishSoundDuration_s, 1, 1)*S.GUI.IncorrectSoundVolume_percent; % white noise punish sound
IncorrectSound = ApplySoundEnvelope(IncorrectSound, Envelope);

EarlyPressPunishSound = GenerateWhiteNoise(SF, S.GUI.EarlyPressPunishSoundDuration_s, 1, 1)*S.GUI.EarlyPressPunishSoundVolume_percent; % white noise punish sound
EarlyPressPunishSound = ApplySoundEnvelope(EarlyPressPunishSound, Envelope);

% generate audio stim same duration as vis gratings
AudioStimSound = GenerateSineWave(SF, S.GUI.AudioStimFreq_Hz, S.GUI.GratingDur_s)*S.GUI.AudioStimVolume_percent; % Sampling freq (hz), Sine frequency (hz), duration (s)
AudioStimSound = ApplySoundEnvelope(AudioStimSound, Envelope);

switch BpodSystem.Data.RigName
    case {'JoystickRig3', 'JoystickRig4'}
        disp('Setup Sound Card Audio...');
        BpodSystem.PluginObjects.Sound.load(13, IncorrectSound);
        BpodSystem.PluginObjects.Sound.load(14, EarlyPressPunishSound);            
    otherwise
        disp('Setup Hifi Audio...');
        H.DigitalAttenuation_dB = -35; % Set a comfortable listening level for most headphones (useful during protocol dev).
        H.load(3, IncorrectSound);
        H.load(4, EarlyPressPunishSound);
end    

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
    case 'JoystickRig1'
        MonitorID = 1;
    case 'JoystickRig2'
        MonitorID = 1;   
    case 'JoystickRig3'
        MonitorID = 2;   
    case 'JoystickRig4'
        MonitorID = 2;             
end    

BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(MonitorID, 0, [0 0], [180 180], 0); % Assumes second monitor is screen #2. Sync patch = 180x180 pixels

BpodSystem.PluginObjects.V.SyncPatchIntensity = 255; % increased, seems 140 doesn't always trigger BNC high

% Indicate loading
BpodSystem.PluginObjects.V.loadText(1, 'Loading...', '', 80);
BpodSystem.PluginObjects.V.play(1);

Ysize = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
Xsize = BpodSystem.PluginObjects.V.ViewPortDimensions(1);

ImgParams.spatialFreq = .005;
ImgParams.orientation = 0;
ImgParams.contrast = 1;
ImgParams.phase = 0.5;
[VideoGrating, VideoGrayFixed] = m_AVstim.GenStimImg(ImgParams, Xsize, Ysize);

% Query duration of one monitor refresh interval:
ifi=Screen('GetFlipInterval', BpodSystem.PluginObjects.V.Window); % check via psychtoolbox
FramesPerSecond = BpodSystem.PluginObjects.V.DetectedFrameRate;

BpodSystem.Data.InterFrameIntervalPsychToolbox = ifi;
BpodSystem.Data.FramesPerSecondVidPlugin = FramesPerSecond;

GratingFrames = m_AVstim.GetFrames(FramesPerSecond, S.GUI.GratingDur_s);

BpodSystem.PluginObjects.V.loadVideo(1, VideoGrating);
BpodSystem.PluginObjects.V.loadVideo(2, VideoGrayFixed);

% compose grating video
GratingFrame_SyncW = BpodSystem.PluginObjects.V.Videos{1}.Data(1);
GratingPattern = [GratingFrame_SyncW GratingFrame_SyncW];
GratingVideo = [repmat(GratingPattern, 1, GratingFrames/2)];
    
% compose gray video, fixed ISI
GrayFrame_SyncW = BpodSystem.PluginObjects.V.Videos{2}.Data(1);
GrayFrame_SyncBlk = BpodSystem.PluginObjects.V.Videos{2}.Data(2);
GrayProbePattern =  [GrayFrame_SyncW GrayFrame_SyncW];

% gray probe video same duration as grating video
ProbeGrayVideo = [repmat(GrayProbePattern, 1, GratingFrames/2)];

% update durations based on number of frames generated
GratingDur = m_AVstim.GetVideoDur(FramesPerSecond, GratingVideo);
    
% use init video to set Frame2TTL BNC sync to be low and not miss first frame of vis
% stim later
GrayInitBNCSync = [repmat(GrayFrame_SyncW, 1, 120) GrayFrame_SyncBlk];
BpodSystem.PluginObjects.V.Videos{6} = struct;
BpodSystem.PluginObjects.V.Videos{6}.nFrames = 121; % + 1 for final frame
BpodSystem.PluginObjects.V.Videos{6}.Data = GrayInitBNCSync;
   
LastGratingDuration = S.GUI.GratingDur_s; % Remember value of stim dur so that we only regenerate the grating video if parameter has changed

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

ProbeTrialTypes = m_TrialConfig.GenProbeTrials(S, TrialTypes, currentTrial);
[OptoTrialTypes, BpodSystem] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, TrialTypes, currentTrial, 1);
m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 0);

%% Vis 2 Jitter
% create arrays for vis 2 jitter
BpodSystem.Data.Vis2Jitter = [];
Vis2Jitter = zeros(1, length(TrialTypes));

% check gui press delay values
% set long delay to be at    least MinShortLongDelaySeparation more than short delay
MinShortLongDelaySeparation = 0.200;
if S.GUI.PrePress2DelayLong_s <= S.GUI.PrePress2DelayShort_s + MinShortLongDelaySeparation
    S.GUI.PrePress2DelayLong_s = S.GUI.PrePress2DelayShort_s + MinShortLongDelaySeparation;
end

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

BpodSystem.Data.EndOfTrialITI = 0;

% init PreReward delays
PreRewardDelay_s = 0;
AutoPreRewardDelay_s = PreRewardDelay_s;
PreviousEnableAutoPreRewardDelay= 0;  % start zero in case enabled by default, then still get start value for delay

% init press delays
PressVisDelay_s = 0;

% check if opto session
% 1 = 'Opto', 2 = 'Control'
if S.GUI.SessionType == 2
    m_Opto.EnableOpto = 0;
else
    m_Opto.EnableOpto = 1;
end


%% Main trial loop

PreviousBlockStart = 1;
StartOfBlock = [1, abs(diff(TrialTypes))];
starts = find(StartOfBlock>0);
next_starts = find(starts>PreviousBlockStart);
LenBlock = next_starts(1)-PreviousBlockStart;
TempOutcome = [];
OptoBlockTag = 2 ;
OptoTrailEpoch = [] ;
stableblock = 0;



for currentTrial = 1:MaxTrials
   
    BpodSystem.Data.CurrentTrial = currentTrial;    % update these later to only use the bpod system struct
    ExperimenterTrialInfo.TrialNumber = currentTrial;   % check variable states as field/value struct for experimenter info
    EnableBlockChange = S.GUI.EnableBlockChange;
    MaxNumTrialsPerBlock = S.GUI.MaxNumTrialsPerBlock;
    % check the varient block size
     
    if EnableBlockChange

        if currentTrial > S.GUI.NumEasyWarmupTrials
            TempOutcome = [TempOutcome , ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1))];
        end
        if length(TempOutcome)>10
            TempOutcome = TempOutcome(2:11); 
        end
        if (StartOfBlock(currentTrial)&&(currentTrial>1))
            if stableblock == 0
                if (sum(TempOutcome) < 5)&&(LenBlock<MaxNumTrialsPerBlock)
                    TrialTypes = [TrialTypes(1:currentTrial-1) , TrialTypes(currentTrial-1) , TrialTypes(currentTrial:end-1)] ;
                    StartOfBlock = [1, abs(diff(TrialTypes))];
                    LenBlock = currentTrial+1-PreviousBlockStart;
                elseif (sum(TempOutcome) >= 5)&&(LenBlock<MaxNumTrialsPerBlock)
                    jitter_additional_trials = 0;
                    previous_outcome = TempOutcome(end);
                    TempOutcome = [];
                    
                    if previous_outcome
                        stableblock = 1;
                        jitter_additional_trials = randperm(S.GUI.MaxJitterRange-S.GUI.MinJitterRange+1) + S.GUI.MinJitterRange-1;
                        jitter_additional_trials = jitter_additional_trials(1) ;
                        PreviousBlockStart = currentTrial+ jitter_additional_trials;
                        for jitter = 1:jitter_additional_trials
                            temp_trial = currentTrial-1 + jitter -1;
                            TrialTypes = [TrialTypes(1:temp_trial) , TrialTypes(temp_trial) , TrialTypes(temp_trial+1:end-1)] ;
                        end
                        StartOfBlock = [1, abs(diff(TrialTypes))];
                    end
                    if (S.GUI.OptoTrialTypeSeq == 6)&&(m_Opto.EnableOpto)
                        if (OptoBlockTag > 1)
                            OptoTrailEpoch = randperm(S.GUI.EpochTrialStop, int8(S.GUI.EpochTrialStop*S.GUI.OnFraction))+currentTrial-1 +jitter_additional_trials;
                            OptoTrialTypes(OptoTrailEpoch) = 2 ;
                        end
                        OptoBlockTag = mod(OptoBlockTag+1 , 4) ;
                    end
                    starts = find(StartOfBlock>0);
                    next_starts = find(starts>PreviousBlockStart);
                    LenBlock = next_starts(1)-PreviousBlockStart;

                elseif (LenBlock>=MaxNumTrialsPerBlock)
                    TempOutcome = [];
                    PreviousBlockStart = currentTrial;
                    if (S.GUI.OptoTrialTypeSeq == 6)&&(m_Opto.EnableOpto)
                        if (OptoBlockTag > 1)
                            OptoTrailEpoch = randperm(S.GUI.EpochTrialStop, int8(S.GUI.EpochTrialStop*S.GUI.OnFraction))+currentTrial-1 ;
                            OptoTrialTypes(OptoTrailEpoch) = 2 ;
                        end
                        OptoBlockTag = mod(OptoBlockTag+1 , 4) ;
                    end
                    starts = find(StartOfBlock>0);
                    next_starts = find(starts>PreviousBlockStart);
                    LenBlock = next_starts(1)-PreviousBlockStart;
                end
            else
                stableblock = 0;
            end
        end
    else
       if (S.GUI.OptoTrialTypeSeq == 6)&&(m_Opto.EnableOpto)
           if (StartOfBlock(currentTrial)&&(currentTrial>1)) && (OptoBlockTag > 1)
               OptoTrailEpoch = randperm(S.GUI.EpochTrialStop, int8(S.GUI.EpochTrialStop*S.GUI.OnFraction))+currentTrial ;
               OptoTrialTypes(OptoTrailEpoch) = 2 ;
           end
           OptoBlockTag = mod(OptoBlockTag+1 , 4) ;
           OptoTrailEpoch = [] ;
       end
    end
    disp(['last 10 outcomes: ' num2str(TempOutcome)]);
    disp(['current block len: ' num2str(LenBlock)]);
    disp(['max block len: ' num2str(MaxNumTrialsPerBlock)]);
    if ~isempty(OptoTrailEpoch) 
        disp(['epoch opto: ' num2str(OptoTrailEpoch)]);
    end

%% sync trial-specific parameters from GUI

    S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
    S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin 

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

    if updateTrialTypeSequence
        TrialTypes = m_TrialConfig.GenTrials(S, MaxTrials, numTrialTypes, TrialTypes, currentTrial, updateTrialTypeSequence);
        ProbeTrialTypes = m_TrialConfig.GenProbeTrials(S, TrialTypes, currentTrial);
        StartOfBlock = [1, abs(diff(TrialTypes))];
    end

    %% update probe trial Enable
    % no probe trials if Probe Enable off
    if ~S.GUI.EnableProbe
        ProbeTrialTypes(currentTrial) = 0;
    end
    BpodSystem.Data.ProbeTrial(currentTrial) = ProbeTrialTypes(currentTrial);
    %% update probe trial types start of block
    % no probe trials in the initial epoch of each block
    if (StartOfBlock(currentTrial)&&(currentTrial>1))
        if TrialTypes(currentTrial) == 1
            epochStop = min(S.GUI.MaxTrials, currentTrial + S.GUI.FirstEpochShortLen);
        else
            epochStop = min(S.GUI.MaxTrials, currentTrial + S.GUI.FirstEpochLongLen);
        end
        ProbeTrialTypes(currentTrial:epochStop) = 0;
    end
    BpodSystem.Data.ProbeTrial(currentTrial) = ProbeTrialTypes(currentTrial);
    
    %% update probe trial types warmup
    % no probe trials during warmup
    if WarmupTrialsCounter > 0
        ProbeTrialTypes(currentTrial) = 0;
    end
    BpodSystem.Data.ProbeTrial(currentTrial) = ProbeTrialTypes(currentTrial);

    %% update opto epoch warmup
    if (S.GUI.OptoTrialTypeSeq >= 5 && ...
        WarmupTrialsCounter > 0)
        OptoTrialTypes(currentTrial) = 1;
    end        
  
    %% update outcome plot

    % trial type updating needs to be updated with addition of probe
    % trials, epoch opto, block margins, etc before being used during session
    m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 0);


    %% update grating and gray videos
    
    gratingChanged = S.GUI.GratingDur_s ~= LastGratingDuration;

    if gratingChanged
        GratingFrames = m_AVstim.GetFrames(FramesPerSecond, S.GUI.GratingDur_s);
        
        % compose grating video
        GratingVideo = [repmat(GratingPattern, 1, GratingFrames/2)];
        ProbeGrayVideo = [repmat(GrayProbePattern, 1, GratingFrames/2)];
        
        % update durations based on number of frames generated
        GratingDur = m_AVstim.GetVideoDur(FramesPerSecond, GratingVideo);
        LastGratingDuration = S.GUI.GratingDur_s;
    end


    %% update video & audio and change tracking variables for audio and vis stim
 
    % if vis stim dur, audio stim freq, or volume changed then update sound wave
    if gratingChanged || ...
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
        IncorrectSound = GenerateWhiteNoise(SF, S.GUI.PunishSoundDuration_s, 1, 1)*S.GUI.IncorrectSoundVolume_percent; % white noise punish sound
        IncorrectSound = ApplySoundEnvelope(IncorrectSound, Envelope);
        switch BpodSystem.Data.RigName
            case {'JoystickRig3', 'JoystickRig4'}
                disp('Setup Sound Card Audio...');
                BpodSystem.PluginObjects.Sound.load(13, IncorrectSound);            
            otherwise
                disp('Setup Hifi Audio...');
                H.load(3, IncorrectSound);
        end    
        LastPunishSoundDuration = S.GUI.PunishSoundDuration_s;
        LastIncorrectSoundVolume = S.GUI.IncorrectSoundVolume_percent;
    end

    if (S.GUI.EarlyPressPunishSoundDuration_s ~= LastEarlyPressPunishSoundDuration) || ...
        (S.GUI.EarlyPressPunishSoundVolume_percent ~= LastEarlyPressPunishSoundVolume)
        EarlyPressPunishSound = GenerateWhiteNoise(SF, S.GUI.EarlyPressPunishSoundDuration_s, 1, 1)*S.GUI.EarlyPressPunishSoundVolume_percent; % white noise punish sound
        EarlyPressPunishSound = ApplySoundEnvelope(EarlyPressPunishSound, Envelope);
        switch BpodSystem.Data.RigName
            case {'JoystickRig3', 'JoystickRig4'}
                disp('Setup Sound Card Audio...');
                BpodSystem.PluginObjects.Sound.load(14, EarlyPressPunishSound);            
            otherwise
                disp('Setup Hifi Audio...');
                H.load(4, EarlyPressPunishSound);
        end                  
        LastEarlyPressPunishSoundDuration = S.GUI.EarlyPressPunishSoundDuration_s;
        LastEarlyPressPunishSoundVolume = S.GUI.EarlyPressPunishSoundVolume_percent;        
    end

    m_Opto.EnableOpto = S.GUI.SessionType ~= 2;
    [OptoTrialTypes, BpodSystem] = m_Opto.UpdateOptoTrials(BpodSystem, S, OptoTrialTypes, TrialTypes, currentTrial, 0);
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

    % load regular video
    BpodSystem.PluginObjects.V.Videos{5} = struct;
    BpodSystem.PluginObjects.V.Videos{5}.nFrames = FullOptoVideoFrames; 
    BpodSystem.PluginObjects.V.Videos{5}.Data = FullOptoVideo;      

    % load gray probe trial video
    BpodSystem.PluginObjects.V.Videos{3} = struct;
    BpodSystem.PluginObjects.V.Videos{3}.nFrames = FullOptoVideoFrames; 
    BpodSystem.PluginObjects.V.Videos{3}.Data = FullOptoProbeVideo;             
          
    % audio for opto delay shift
    OptoAudioStimOffsetNumSamples = floor(VideoOptoDelayDur * SF); % get duration of gray opto delay in number of audio samples for period between audio stim 

    OptoAudioStimOffset = zeros(1, OptoAudioStimOffsetNumSamples);

    OptoAudioStimSound = [OptoAudioStimOffset AudioStimSound];


    %% update trial-specific valve times using calibration table according to set reward amount    
    CenterValveTime = 0;
    if ~ProbeTrialTypes(currentTrial)
        if S.GUI.EnableAlternatingReward == 0
            RewardAmount_uL = S.GUI.CenterValveAmount_uL;
        else
            if TrialTypes(currentTrial) == 1
                RewardAmount_uL = S.GUI.ShortRewardAmount_uL;
            else
                RewardAmount_uL = S.GUI.LongRewardAmount_uL;
            end
        end
        CenterValveTime = GetValveTimes(RewardAmount_uL, [2]); 
    end
    % init reward times, update based on reward rep
    RewardTime = CenterValveTime;

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
                if (S.GUI.PrePress2DelayLong_s >= MinShortLongDelaySeparation)
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


     
    %% get press window

    switch TrialTypes(currentTrial)
        case 1
            if  (currentTrial>1 && ...      
                S.GUI.EnableAutoPressWinReduce && ...
                (WarmupTrialsCounter <= 0) && ...
                ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1)))
                S.GUI.Press1Window_s = max(S.GUI.Press1Window_s - S.GUI.AutoPressWinReduceStep, S.GUI.AutoPressWin1ReduceMin);
                S.GUI.Press2WindowShort_s = max(S.GUI.Press2WindowShort_s - S.GUI.AutoPressWinReduceStep, S.GUI.AutoPressWinShortReduceMin);
            end
    
            % local vars for checking if warmup extend
            Press1Window_s = S.GUI.Press1Window_s;
            Press2Window_s = S.GUI.Press2WindowShort_s;
        case 2
            if  (currentTrial>1 && ...      
                S.GUI.EnableAutoPressWinReduce && ...
                (WarmupTrialsCounter <= 0) && ...
                ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1)))
                S.GUI.Press1Window_s = max(S.GUI.Press1Window_s - S.GUI.AutoPressWinReduceStep, S.GUI.AutoPressWin1ReduceMin);
                S.GUI.Press2WindowLong_s = max(S.GUI.Press2WindowLong_s - S.GUI.AutoPressWinReduceStep, S.GUI.AutoPressWinLongReduceMin);
            end
    
            % local vars for checking if warmup extend
            Press1Window_s = S.GUI.Press1Window_s;
            Press2Window_s = S.GUI.Press2WindowLong_s;
    end

    %% fixed dur Pre-Vis ITI
    PreVisStimITI = S.GUI.ITI_Pre; % updated V_3_3; updated V_3_7            
     
    %% Draw trial-specific ITI post for end of trial ITI    
    BpodSystem.Data.EndOfTrialITI = m_TrialConfig.GetITI(S); % updated V_3_3; updated V_3_7

    %% set state matrix variables        
    
    VisDetectGray1OutputAction = {'RotaryEncoder1', ['E']};

    WaitForPress1_OutputActions = {'SoftCode', 7,'RotaryEncoder1', ['E']};
    Press1_OutputActions = {'RotaryEncoder1', ['E']};
    PreRetract1Delay_OutputActions = {};
    LeverRetract1_OutputActions = {'SoftCode', 8};
    DidNotPress1_OutputActions = {};
    EarlyPress1_OutputActions = {};        

    PreDelayGap_OutputActions = {};
    WaitForPress2_OutputActions = {'SoftCode', 7,'RotaryEncoder1', ['E']};             
    Press2_OutputActions = {'RotaryEncoder1', ['E']};
    DidNotPress2_OutputActions = {};
    EarlyPress2_OutputActions = {};
       
    PreRewardDelay_OutputActions = {};

    Punish_OutputActions = {};

    if ProbeTrialTypes(currentTrial)
        Reward_OutputActions = {};
    else
        Reward_OutputActions = {'Valve2', 1};
    end

    ITI_OutputActions = {'SoftCode', 8, 'GlobalCounterReset', '00000111111111'}; 
    Punish_ITI_OutputActions = {} ;
    EarlyPress2Punish_OutputActions = {} ;

    %% Opto timers

    % Define Opto Timer Trigger and Cancel for Segment 1 and Segment 2
    % seg1:
    % vis1 and/or wait1: timer 1 and 5: '000010001'
    TimerTrigger_V1W1 = {'GlobalTimerTrig', '000010001'};

    % seg delay:
    % vis1 and/or wait1: timer 4 and 6: '000101000'
    TimerTrigger_PressDelay = {'GlobalTimerTrig', '000101000'};
    TimerCancel_PressDelay = {'GlobalTimerCancel', '000101000'};

    % seg2:
    % vis2 and/or wait2: timer 3 and 7: '001000100'
    TimerTrigger_V2W2 = {'GlobalTimerTrig', '001000100'};

    % shutter reset timer: timer 2: '000000010'
    TimerShutterReset = {'GlobalTimerTrig', '000000010'};

    % press timers: timer 8 and 9: 
    % currently using seg delay timers 4 & 6 since press states have
    % unknown immediate start time of opto, update to use different
    % timers if seg delay changes
    
    TimerTrigger_Press = {'GlobalTimerTrig', '110000000'};

    %% update trial-specific Audio
    EarlyPressPunish_OutputActions = {};

    % set visual stimulus 1&2 state outputs (referred to as audStimOpto
    % since the audio starts in those states)

    % set audio stim based on audio enable
    switch BpodSystem.Data.RigName
        case {'JoystickRig3', 'JoystickRig4'}
            disp('Setup Sound Card Audio...');
            BpodSystem.PluginObjects.Sound.load(15, OptoAudioStimSound);
            if S.GUI.AudioStimEnable
                AudStim = {'SoftCode', 15};
            else
                AudStim = {};
            end      
            % toggle punish sound on/off
            if (S.GUI.IncorrectSound && ...
               ~ProbeTrialTypes(currentTrial))
                    Punish_OutputActions = [Punish_OutputActions, 'SoftCode', 13];
                    EarlyPressPunish_OutputActions = {'SoftCode', 14};          
            end                      
        otherwise
            disp('Setup Hifi Audio...');
            H.load(7, OptoAudioStimSound);
            if S.GUI.AudioStimEnable
                AudStim = {'HiFi1', ['P', 6]};
            else
                AudStim = {};
            end
            % toggle punish sound on/off
            if (S.GUI.IncorrectSound && ...
               ~ProbeTrialTypes(currentTrial))
                    Punish_OutputActions = [Punish_OutputActions, 'HiFi1', ['P' 2]];
                    EarlyPressPunish_OutputActions = {'HiFi1', ['P' 3]}; 
                    EarlyPress2Punish_OutputActions = {'HiFi1', ['P' 3]};
            end                      
    end            
    
    % set visual stimulus output actions
    % add audio output to vis stim state output
    VisualStimulus1_OutputActions = AudStim;

    % if opto is enabled, add opto triggers to vis stim output actions
    % update after opto proto is defined to create separate function to abstract global timer
    if m_Opto.EnableOpto && (OptoTrialTypes(currentTrial) == 2)
        %%%%%%%%%%%%%%% segment either %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        %%%%%%%%%%%%%%% segment 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % if any segment 1 opto is active,
        % cancel timers and reset shutter when press1, didnotpress1, earlypress1 starts            
        if S.GUI.OptoVis1 || S.GUI.OptoWaitForPress1 || S.GUI.OptoPress1            
            if S.GUI.OptoPress1
                PreRetract1Delay_OutputActions = [Press1_OutputActions, TimerShutterReset, {'GlobalTimerCancel', '111111101'}];
            else
                Press1_OutputActions = [Press1_OutputActions, TimerShutterReset, {'GlobalTimerCancel', '111111101'}];
            end
            DidNotPress1_OutputActions = [DidNotPress1_OutputActions, TimerShutterReset, {'GlobalTimerCancel', '111111101'}];
            EarlyPress1_OutputActions = [EarlyPress1_OutputActions, TimerShutterReset, {'GlobalTimerCancel', '111111101'}];
        end

        if S.GUI.OptoVis1
            VisDetectGray1OutputAction = [VisDetectGray1OutputAction , TimerTrigger_V1W1];
        end                       

        if S.GUI.OptoVis1 && ~S.GUI.OptoWaitForPress1
            %WaitForPress1_OutputActions = [WaitForPress1_OutputActions, TimerCancel_V1W1];
        end

        if ~S.GUI.OptoVis1 && S.GUI.OptoWaitForPress1
            VisualStimulus1_OutputActions = [VisualStimulus1_OutputActions, TimerTrigger_V1W1];
        end

        if ~S.GUI.OptoWaitForPress1 && S.GUI.OptoPress1
            Press1_OutputActions = [Press1_OutputActions, TimerTrigger_Press];
        end

        %%%%%%%%%%%%%%% segment 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % if any segment 2 (including prepressdelay) opto is active,
        % cancel timers and reset shutter when press2, didnotpress2, earlypress2 starts
        if S.GUI.OptoPrePressDelay || S.GUI.OptoVis2 || S.GUI.OptoWaitForPress2 || S.GUI.OptoPress2
            if S.GUI.OptoPress2
                PreRewardDelay_OutputActions = [PreRewardDelay_OutputActions, TimerShutterReset, {'GlobalTimerCancel', '111111101'}];
            else
                Press2_OutputActions = [Press2_OutputActions, TimerShutterReset, {'GlobalTimerCancel', '111111101'}];
            end                                   
            DidNotPress2_OutputActions = [DidNotPress2_OutputActions, TimerShutterReset, {'GlobalTimerCancel', '111111101'}];
            EarlyPress2_OutputActions = [EarlyPress2_OutputActions, TimerShutterReset, {'GlobalTimerCancel', '111111101'}];
        end

        % cases depending on PrePressDelay opto on/off
        if S.GUI.OptoPrePressDelay
            PreDelayGap_OutputActions = [PreDelayGap_OutputActions, TimerTrigger_PressDelay];


            if ~S.GUI.OptoWaitForPress2
                WaitForPress2_OutputActions = [WaitForPress2_OutputActions, TimerCancel_PressDelay];
                if S.GUI.OptoPress2
                    Press2_OutputActions = [Press2_OutputActions, TimerTrigger_Press];
                end
            end
                          
        else

            if S.GUI.OptoWaitForPress2 && S.GUI.SelfTimedMode
                PreDelayGap_OutputActions = [PreDelayGap_OutputActions, TimerTrigger_V2W2];
            end   

            if ~S.GUI.OptoWaitForPress2
                if S.GUI.OptoPress2
                    Press2_OutputActions = [Press2_OutputActions, TimerTrigger_Press];
                end
            end
        end

        %%%%%%%%%%%%%%% ITI %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if S.GUI.OptoRewardITI
            %ITI_OutputActions = [ITI_OutputActions, {'GlobalTimerTrig', '10000000000000'}];
            Reward_OutputActions = [Reward_OutputActions, {'GlobalTimerTrig', '10000000000000'}];
        end
        if S.GUI.OptoEarlyPressITI
            EarlyPress2Punish_OutputActions = [EarlyPress2Punish_OutputActions, {'GlobalTimerTrig', '10000000000000'}];
        end

        % need the punish shutter retract?
        % Punish_OutputActions = [Punish_OutputActions, TimerShutterReset];
    end

    %% update trial-specific Audio

    LeverRetractFinal_StateChangeConditions = {'SoftCode2', 'ITI_Switch'};
    WaitForPress1_StateChangeConditions = {'Tup', 'DidNotPress1', 'RotaryEncoder1_3', 'Press1'};
    LeverRetract1_StateChangeConditions = {'SoftCode2', 'PreDelayGap'};
    
    WaitForPress2_StateChangeConditions = {'Tup', 'DidNotPress2', 'RotaryEncoder1_3', 'Press2'};             
               
    %% adjust for warmup trials
    % For warmup trials, wait for press is extended by additional warmup param, after warmup wait for press is S.GUI.PressWindow_s
    
    % check if user has changed number of warmup trials    
    if S.GUI.NumEasyWarmupTrials ~= LastNumEasyWarmupTrials
        WarmupTrialsCounter = S.GUI.NumEasyWarmupTrials;    % update warmup trial counter to current gui param
        LastNumEasyWarmupTrials = S.GUI.NumEasyWarmupTrials;    % store current value to check for change again
    end
                   
    if WarmupTrialsCounter > 0

        PressVisDelay_s = min(S.GUI.PrePress2DelayShort_s, PressVisDelay_s);

        Threshold = S.GUI.WarmupThreshold;

        Press2Window_s = S.GUI.Press2WindowWarmup_s;

        BpodSystem.Data.IsWarmupTrial(currentTrial) = 1;            

    else    
        Threshold = S.GUI.Threshold;

        BpodSystem.Data.IsWarmupTrial(currentTrial) = 0;

    end

    BpodSystem.Data.PressThresholdUsed(currentTrial) = Threshold;

    % decrement
    if WarmupTrialsCounter > 0
        WarmupTrialsCounter = WarmupTrialsCounter - 1;
    end
 

    %% update encoder threshold from params

    DetectPressThreshold = S.GUI.RetractThreshold + 0.1;

    BpodSystem.PluginObjects.R.stopUSBStream;   % stop USB streaming to update encoder params
    pause(0.05);
    % BpodSystem.PluginObjects.R.thresholds = [Threshold S.GUI.EarlyPressThreshold];    % udate threshold from GUI params
    BpodSystem.PluginObjects.R.thresholds = [Threshold S.GUI.EarlyPressThreshold DetectPressThreshold];    % udate threshold from GUI params
    BpodSystem.PluginObjects.R.startUSBStream;  % restart encoder USB streaming

    BpodSystem.Data.TrialData{1, S.GUI.currentTrial}.LeverResetPos = []; % array for lever reset positions
                          
    %% add console print for experimenter trial information, these vars are here to make them easier to see when printed on console
    
    ExperimenterTrialInfo.SessionType = OptoStateExpInfo;
    ExperimenterTrialInfo.OptoTrial = OptoTrialExpInfo;  
    ExperimenterTrialInfo.Threshold = S.GUI.Threshold;
    ExperimenterTrialInfo.EarlyPressThreshold = S.GUI.EarlyPressThreshold;
    ExperimenterTrialInfo.Press1Window_s = S.GUI.Press1Window_s;
    ExperimenterTrialInfo.Press2WindowShort_s = S.GUI.Press2WindowShort_s;
    ExperimenterTrialInfo.Press2WindowLong_s = S.GUI.Press2WindowLong_s;
    ExperimenterTrialInfo.PrePress2Delay_s = PressVisDelay_s;
    ExperimenterTrialInfo.PrePress2DelayShort_s = S.GUI.PrePress2DelayShort_s;
    ExperimenterTrialInfo.PrePress2DelayLong_s = S.GUI.PrePress2DelayLong_s;

    strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
    disp(strExperimenterTrialInfo);          

    %% construct state matrix

    sma = NewStateMatrix(); % Assemble state matrix
    
    sma = m_Opto.InsertGlobalTimer(BpodSystem, sma, S, VisStim, PressVisDelay_s);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % adding variables to improve readability, starting with V_3_8
    % output action variables
    StartSyncSignal = {'BNC1', 1};
    RotaryEncoderStart = {'RotaryEncoder1', ['E#' 0]}; % enable rotary encoder and set time sync

    % set audio stim based on audio enable
    switch BpodSystem.Data.RigName
        case {'JoystickRig3', 'JoystickRig4'}
            disp('Setup Sound Card Audio...');
            AudioStart = {};
        otherwise
            disp('Setup Hifi Audio...');
            AudioStart = {['' 'HiFi1'],'*'}; % push newly uploaded waves to front (playback) buffers
    end  


    Start_OutputActions = [AudioStart, RotaryEncoderStart, StartSyncSignal];

    BpodSystem.Data.IsRewardPulse(currentTrial) = 0;

    sma = AddState(sma, 'Name', 'Start', ...
        'Timer', 0.068,...
        'StateChangeConditions', {'Tup', 'PreVisStimITI', 'RotaryEncoder1_2', 'EarlyPress'},...
        'OutputActions', Start_OutputActions); % Code to
    
    sma = AddState(sma, 'Name', 'PreVisStimITI', ...
        'Timer', PreVisStimITI,...
        'StateChangeConditions', {'Tup', 'VisDetect1', 'RotaryEncoder1_2', 'EarlyPress'},...
        'OutputActions', {});
    
    %% rep 1
    
    sma = AddState(sma, 'Name', 'VisDetect1', ...
        'Timer', 0.100,...
        'StateChangeConditions', {'Tup', 'VisStimInterruptDetect1', 'BNC1High', 'VisDetectGray1', 'RotaryEncoder1_2', 'EarlyPress1'},...
        'OutputActions', {'SoftCode', 5,'RotaryEncoder1', ['E']});

    % VisDetectGray1OutputAction = {‘RotaryEncoder1’, [‘E’]};
    sma = AddState(sma, 'Name', 'VisDetectGray1', ...
        'Timer', 0.050,...
        'StateChangeConditions', {'Tup', 'VisStimInterruptGray1', 'BNC1High', 'VisualStimulus1', 'RotaryEncoder1_2', 'EarlyPress1'},...
        'OutputActions', VisDetectGray1OutputAction);        

    % VisualStimulus1_OutputActions = [AudStim, TimerTrigger_V1W1, 'RotaryEncoder1', ['E']]        
    sma = AddState(sma, 'Name', 'VisualStimulus1', ...
        'Timer', VisStim.VisStimDuration + 0.020,...
        'StateChangeConditions', {'BNC1Low', 'WaitForPress1'},...
        'OutputActions', [VisualStimulus1_OutputActions, 'RotaryEncoder1', ['E']]);

    % Arguments: (sma, GlobalTimerNumber, Duration(s))
    sma = SetGlobalTimer(sma, 10, Press1Window_s); % Press1 window timer % global timer 10 is triggered when we enter waitForPress1; its duration is Press1Window
    WaitForPress1_OutputActions = [WaitForPress1_OutputActions, 'GlobalTimerTrig', 10];        
    % WaitForPress1_StateChangeConditions = {'Tup', 'DidNotPress1', 'RotaryEncoder1_3', 'Press1'};
    % WaitForPress1_OutputActions = {'SoftCode', 7,'RotaryEncoder1', ['E'] , 'GlobalTimerTrig', 10};
    sma = AddState(sma, 'Name', 'WaitForPress1', ...
        'Timer', Press1Window_s,...
        'StateChangeConditions', WaitForPress1_StateChangeConditions,...
        'OutputActions', WaitForPress1_OutputActions);

    % Press1_OutputActions ={'RotaryEncoder1', ['E']} and opto
    sma = AddState(sma, 'Name', 'Press1', ...
        'Timer', Press1Window_s,...
        'StateChangeConditions', {'Tup', 'DidNotPress1', 'GlobalTimer10_End', 'DidNotPress1', 'RotaryEncoder1_1', 'PreRetract1Delay'},...
        'OutputActions', Press1_OutputActions);

    sma = AddState(sma, 'Name', 'PreRetract1Delay', ...
        'Timer', 0.100,...
        'StateChangeConditions', {'Tup', 'LeverRetract1'},...
        'OutputActions', PreRetract1Delay_OutputActions);             
    
    sma = AddState(sma, 'Name', 'LeverRetract1', ...
        'Timer', 0,...
        'StateChangeConditions', LeverRetract1_StateChangeConditions,... % When the PC is done resetting the lever, it sends soft code 1 to the state machine
        'OutputActions', LeverRetract1_OutputActions); % On entering the LeverRetract state, send soft code 1 to the PC. The soft code handler will then start resetting the lever.   

    sma = AddState(sma, 'Name', 'PreDelayGap', ...
            'Timer', 0.05,... %%0.0221
            'StateChangeConditions', {'Tup', 'PrePress2Delay'},...
            'OutputActions', PreDelayGap_OutputActions); 

    if currentTrial>2 && S.GUI.AssistMode==1 && rand<S.GUI.AssistProb && ...
       isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'EarlyPress2Punish') && ...
       ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.EarlyPress2Punish(1))

        BpodSystem.Data.AssistTrial(currentTrial) = 1;

        sma = AddState(sma, 'Name', 'PrePress2Delay', ...
            'Timer', PressVisDelay_s,...
            'StateChangeConditions', {'Tup', 'Assist'},...
            'OutputActions', {});

        sma = AddState(sma, 'Name', 'Assist', ...
            'Timer', 0.05,...
            'StateChangeConditions', {'Tup', 'WaitForPress2'},...
            'OutputActions', {'SoftCode', 7, 'RotaryEncoder1', ['E']});

    else

        BpodSystem.Data.AssistTrial(currentTrial) = 0;

        sma = AddState(sma, 'Name', 'PrePress2Delay', ...
            'Timer', PressVisDelay_s,...
            'StateChangeConditions', {'RotaryEncoder1_2', 'EarlyPress2', 'Tup', 'WaitForPress2'},...
            'OutputActions', {'SoftCode', 7, 'RotaryEncoder1', ['E']});
    end

    sma = SetGlobalTimer(sma, 11, Press2Window_s); % Press2 window timer 
    WaitForPress2_OutputActions = [WaitForPress2_OutputActions, 'GlobalTimerTrig', 11];

    sma = AddState(sma, 'Name', 'WaitForPress2', ...
        'Timer', Press2Window_s,...
        'StateChangeConditions', WaitForPress2_StateChangeConditions,...    
        'OutputActions', WaitForPress2_OutputActions);         

    sma = AddState(sma, 'Name', 'Press2', ...
        'Timer', Press2Window_s,...
        'StateChangeConditions', {'Tup', 'DidNotPress2', 'RotaryEncoder1_1', 'RewardLeverRetract'},...
        'OutputActions', Press2_OutputActions);

    sma = AddState(sma, 'Name', 'RewardLeverRetract', ...
        'Timer', 0.1,...
        'StateChangeConditions', {'Tup', 'PreRewardDelay'},...
        'OutputActions', {'SoftCode', 8});

    sma = AddState(sma, 'Name', 'PreRewardDelay', ...
        'Timer' , PreRewardDelay_s, ...
        'StateChangeConditions', {'Tup', 'Reward'}, ...
        'OutputActions', PreRewardDelay_OutputActions);        

    % Reward_OutputActions = {'Valve2', 1}
    sma = AddState(sma, 'Name', 'Reward', ...
        'Timer', RewardTime,...
        'StateChangeConditions', {'Tup', 'PostRewardDelay'},...
        'OutputActions', Reward_OutputActions); 

    % timer to indicate if reward occurred 
    % Arguments: (sma, TimerNumber, TimerDuration)
    sma = SetGlobalTimer(sma, 13, 20); 

    % condition to indicate if timer 13 is active
    % Arguments: (sma, ConditionNumber, ConditionChannel, ConditionValue; 1 = on, 0 = off)
    sma = SetCondition(sma, 1, 'GlobalTimer13', 1);
    
    PostRewardDelay_StateChangeConditions = {'Tup', 'LeverRetractFinal'};
    % trigger global timer 13 to indicate reward occured
    sma = AddState(sma, 'Name', 'PostRewardDelay', ...
        'Timer', S.GUI.PostRewardDelay_s,...
        'StateChangeConditions', PostRewardDelay_StateChangeConditions,...
        'OutputActions', {'GlobalTimerTrig', 13});  

    sma = AddState(sma, 'Name', 'DidNotPress1', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'Punish'},...
        'OutputActions', DidNotPress1_OutputActions);	% {opto timers}
    
    sma = AddState(sma, 'Name', 'DidNotPress2', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'Punish'},...
        'OutputActions', DidNotPress2_OutputActions);	% {opto timers}

    sma = AddState(sma, 'Name', 'EarlyPress', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'EarlyPressPunish'},...
        'OutputActions', {});        

    % EarlyPress1_OutputActions = {TimerShutterReset, {'GlobalTimerCancel', '111111101'}
    sma = AddState(sma, 'Name', 'EarlyPress1', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'EarlyPress1Punish'},...
        'OutputActions', EarlyPress1_OutputActions);

    % EarlyPress2_OutputActions = TimerShutterReset, {'GlobalTimerCancel', '111111101'}
    sma = AddState(sma, 'Name', 'EarlyPress2', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'EarlyPress2LeverRetract'},...
        'OutputActions', EarlyPress2_OutputActions);

    sma = AddState(sma, 'Name', 'EarlyPress2LeverRetract', ...
        'Timer', 0.1,...
        'StateChangeConditions', {'Tup', 'EarlyPress2Punish'},...
        'OutputActions', {'SoftCode', 8});

    % EarlyPressPunish_OutputActions = {'HiFi1', ['P' 3]};
    sma = AddState(sma, 'Name', 'EarlyPressPunish', ...
        'Timer', S.GUI.EarlyPressPunishSoundDuration_s,...
        'StateChangeConditions', {'Tup', 'LeverRetractFinal'},...
        'OutputActions', EarlyPressPunish_OutputActions);  

    % EarlyPressPunish_OutputActions = {'HiFi1', ['P' 3]};
    sma = AddState(sma, 'Name', 'EarlyPress1Punish', ...
        'Timer', S.GUI.EarlyPressPunishSoundDuration_s,...
        'StateChangeConditions', {'Tup', 'LeverRetractFinal'},...
        'OutputActions', EarlyPressPunish_OutputActions);

    % EarlyPressPunish_OutputActions = {'HiFi1', ['P' 3]};
    sma = AddState(sma, 'Name', 'EarlyPress2Punish', ...
        'Timer', S.GUI.EarlyPressPunishSoundDuration_s,...
        'StateChangeConditions', {'Tup', 'LeverRetractFinal'},...
        'OutputActions', EarlyPress2Punish_OutputActions);

    % LeverRetractFinal_StateChangeConditions = {'SoftCode2', 'ITI_Switch'}
    % SoftCode2: Indicate to the state machine that the lever is back in the home position
    sma = AddState(sma, 'Name', 'LeverRetractFinal', ...
        'Timer', 0,...
        'StateChangeConditions', LeverRetractFinal_StateChangeConditions,...
        'OutputActions', {'SoftCode', 8});

    sma = AddState(sma, 'Name', 'ITI_Switch', ...
        'Timer', 0.001,...
        'StateChangeConditions', {'Tup', 'Punish_ITI', 'Condition1', 'ITI'},...
        'OutputActions', {});        

    % Punish_OutputActions = [Punish_OutputActions, TimerShutterReset];
    sma = AddState(sma, 'Name', 'Punish', ...
        'Timer', S.GUI.PunishSoundDuration_s,...
        'StateChangeConditions', {'Tup', 'LeverRetractFinal'},...
        'OutputActions', Punish_OutputActions);         

    sma = AddState(sma, 'Name', 'VisStimInterruptDetect1', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'VisStimInterruptGray1', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});        

    sma = AddState(sma, 'Name', 'VisStimInterruptDetect2', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'VisStimInterruptGray2', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});              

    % S.GUI.PunishITI = 1;        
    sma = AddState(sma, 'Name', 'Punish_ITI', ...
        'Timer', S.GUI.PunishITI,...
        'StateChangeConditions', {'Tup', 'Post_Punish_ITI'},...
        'OutputActions', Punish_ITI_OutputActions);         

    sma = AddState(sma, 'Name', 'Post_Punish_ITI', ...
        'Timer', BpodSystem.Data.EndOfTrialITI,...
        'StateChangeConditions', {'Tup', '>exit'},...
        'OutputActions', {'SoftCode', 8, 'GlobalCounterReset', '00000111111111'});

    sma = AddState(sma, 'Name', 'ITI', ...
        'Timer', BpodSystem.Data.EndOfTrialITI,...
        'StateChangeConditions', {'Tup', '>exit'},...
        'OutputActions', ITI_OutputActions);

    SendStateMachine(sma); % Send the state matrix to the Bpod device   
    RawEvents = RunStateMachine; % Run the trial and return events
   
    if ~isempty(fieldnames(RawEvents)) % If trial data was returned (i.e. if not final trial, interrupted by user)
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
        BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
        BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial); % Adds the trial type of the current trial to data
        BpodSystem.Data.Vis2Jitter(currentTrial) = Vis2Jitter(currentTrial); % Adds the trial type of the current trial to data
        BpodSystem.Data.PrePress2Delay(currentTrial) = PressVisDelay_s;
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, 1);

        SavePlot;

        if useStateTiming
            StateTiming();
        end

        BpodSystem.Data.EncoderData{currentTrial} = BpodSystem.PluginObjects.R.readUSBStream(); % Get rotary encoder data captured since last call to R.readUSBStream()
        % Align this trial's rotary encoder timestamps to state machine trial-start (timestamp of '#' command sent from state machine to encoder module in 'TrialStart' state)
        BpodSystem.Data.EncoderData{currentTrial}.Times = BpodSystem.Data.EncoderData{currentTrial}.Times - BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps(1); % Align timestamps to state machine's trial time 0
        BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps = BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps - BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps(1); % Align event timestamps to state machine's trial time 0
    

        % Update rotary encoder plot
        TrialDuration = BpodSystem.Data.TrialEndTimestamp(currentTrial)-BpodSystem.Data.TrialStartTimestamp(currentTrial);
        if currentTrial == 1
            PreviousEncoderData = [];
        else
            PreviousEncoderData = BpodSystem.Data.EncoderData{currentTrial-1};
        end
        BpodSystem.Data.EncoderData{currentTrial} = CompleteEncoderData( ...
            BpodSystem.Data.EncoderData{currentTrial}, TrialDuration, PreviousEncoderData);

        PreVisStimITITimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.PreVisStimITI;
        VisDetect1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisDetect1;
        VisualStimulus1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.VisualStimulus1;
        WaitForPress1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.WaitForPress1;
        LeverRetract1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.LeverRetract1;
        Reward1Times = [NaN NaN]; % removed rew1 and rew2 V_3_3
        DidNotPress1Times = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.DidNotPress1;
         
        ITITimes = BpodSystem.Data.RawEvents.Trial{1, currentTrial}.States.ITI;

        LeverResetPos = BpodSystem.Data.TrialData{1, currentTrial}.LeverResetPos;

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
            PrePress2DelayTimes, ...
            EarlyPress2Times);

        SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file 


    end

    S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin

    HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.

end

BpodSystem.PluginObjects.V = [];
BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
BpodSystem.PluginObjects.R = [];

% set servo to out position
M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - S.GUI.ServoOutPos));    
M = [];

end

% generate full envelope for sound given the sound and front part of
% envelope, return enveloped sound
function [SoundWithEnvelope] = ApplySoundEnvelope(Sound, Envelope)
    BackOfTheEnvelope = fliplr(Envelope);   % flipe front envelope to get back envelope
    IdxsBetweenTheEnvelope = length(Sound) - 2 * length(Envelope); % indices between front and back of envelope
    FullEnvelope = [Envelope ones(1, IdxsBetweenTheEnvelope) BackOfTheEnvelope];  % full envelope
    SoundWithEnvelope = Sound .* FullEnvelope;    % apply envelope element-wise
end

% match rig ID to computer name for rig-specific settings
% (features/timing/servos/etc)
function BpodSystem = SetRigID(BpodSystem)
    BpodSystem.Data.ComputerHostName = getenv('COMPUTERNAME');
    BpodSystem.Data.RigName = '';
    switch BpodSystem.Data.ComputerHostName
        case 'COS-3A11406'
            BpodSystem.Data.RigName = 'ImagingRig';
        case 'COS-3A11427'
            BpodSystem.Data.RigName = 'JoystickRig1';
        case 'COS-3A17904'
            BpodSystem.Data.RigName = 'JoystickRig2';
        case 'COS-3A14773'
            BpodSystem.Data.RigName = 'JoystickRig3';
        case 'COS-3A14829'
            BpodSystem.Data.RigName = 'JoystickRig4';            
    end
end
