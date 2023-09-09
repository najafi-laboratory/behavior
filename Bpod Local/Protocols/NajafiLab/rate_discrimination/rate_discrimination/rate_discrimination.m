%{
----------------------------------------------------------------------------

Description

----------------------------------------------------------------------------
%}
function rate_discrimination


global BpodSystem

global ShowDebugOutput;
ShowDebugOutput = 0;

global ForceITIZero; % to speed up testing
ForceITIZero = 0;

%% Turn off Bpod LEDs
% This code will disable the state machine status LED
BpodSystem.setStatusLED(0);

%% Assert HiFi module is present + USB-paired (via USB button on console GUI)
BpodSystem.assertModule('HiFi', 1); % The second argument (1) indicates that the HiFi module must be paired with its USB serial port
% Create an instance of the HiFi module
H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1); % The argument is the name of the HiFi module's USB serial port (e.g. COM3)

%% Define parameters
S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S
if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
    %% ITI params
    S.GUI.ITImin_s = 1;    % Minimum ITI (in seconds)
    S.GUI.ITImax_s = 5;    % Maximum ITI (in seconds)
    S.GUI.ITIlambda = 0.3;  % ITIlambda parameter of the exponential distribution
    S.GUIPanels.ITI_Dist = {'ITImin_s', 'ITImax_s', 'ITIlambda'};

    %% sound params
    % S.GUI.DigitalAttenuation_dB = -30; % volume control: H.DigitalAttenuation_dB = -40;
    % S.GUI.MasterVolume_percent = 0.5;    % volume control
    % S.GUIPanels.Sound = {'DigitalAttenuation_dB', 'MasterVolume_percent'};

    %% init cue params
    S.GUI.InitCueVolume_percent = 0.5;  % volume control
    S.GUI.InitCueDuration_s = 0.05; % Duration of init sound
    S.GUI.InitWindowTimeout_s = 5; % How long the mouse has to initiate stimulus or miss init lick
    %S.GUI.InitCueFreq_Hz = 500; % Frequency of init cue
    %S.GUI.InitCueFreq_Hz = 525; % Frequency of init cue, even multiple of 44100 SF
    S.GUI.InitCueFreq_Hz = 4900; % Frequency of init cue, even multiple of 44100 SF
    S.GUIPanels.InitCue = {'InitCueVolume_percent', 'InitCueDuration_s', 'InitWindowTimeout_s', 'InitCueFreq_Hz'};

    %% go cue params
    S.GUI.GoCueVolume_percent = 0.5;  % volume control
    S.GUI.GoCueDuration_s = 0.05; % Duration of go sound
    %S.GUI.GoCueFreq_Hz = 2000; % Frequency of go cue
    %S.GUI.GoCueFreq_Hz = 2100; % Frequency of go cue, even multiple of 44100 SF
    S.GUI.GoCueFreq_Hz = 11025; % Frequency of go cue, even multiple of 44100 SF
    S.GUIPanels.GoCue = {'GoCueVolume_percent', 'GoCueDuration_s', 'GoCueFreq_Hz'};

    %% training level params
    S.GUI.TrainingLevel = 1; % Configurable training and test schemes.
                             % 1 - 'Naive', 2 - 'Trained'
    S.GUIMeta.TrainingLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
    S.GUIMeta.TrainingLevel.String = {'Naive', 'Mid Trained 1', 'Mid Trained 2', 'Well Trained'};
    S.GUI.NumEasyWarmupTrials = 20;
    S.GUIPanels.Training = {'TrainingLevel', 'NumEasyWarmupTrials'};

    %% difficulty params
    % percentage of full perturbation range as boundaries for difficulty levels
    S.GUI.PercentTrialsEasy = 100;
    S.GUI.PercentTrialsMediumEasy = 0;
    S.GUI.PercentTrialsMediumHard = 0;
    S.GUI.PercentTrialsHard = 0;    
    S.GUIPanels.Difficulty = {'PercentTrialsEasy', 'PercentTrialsMediumEasy', 'PercentTrialsMediumHard', 'PercentTrialsHard'};

    %% audio stim
    S.GUI.AudioStimEnable = 1;
    S.GUIMeta.AudioStimEnable.Style = 'checkbox';
    S.GUI.AudioStimVolume_percent = 0.5;  % volume control
    %S.GUI.AudioStimFreq_Hz = 15000; % Frequency of audio stim
    S.GUI.AudioStimFreq_Hz = 14700; % Frequency of audio stim, even multiple of SF = 44100
    S.GUIPanels.AudioStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz'};

    %% vis stim params
    S.GUI.VisStimEnable = 1;
    S.GUIMeta.VisStimEnable.Style = 'checkbox';
    %S.GUI.GratingDur_s = 0.25; % Duration of grating stimulus in seconds - ORIGINAL
    S.GUI.GratingDur_s = 0.1; % Duration of grating stimulus in seconds - UPDATE
    %S.GUI.ISIOrig_s = 0.75; % Duration of *fixed* gray screen stimulus in seconds - ORIGINAL
    S.GUI.ISIOrig_s = 0.5; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
    S.GUI.ExtraStimDurPostRew_Naive_s = 5; % naive mouse sees stimulus for this time (sec) after correct lick    
    S.GUI.NumISIOrigRep = 5; % number of grating/gray repetitions for vis stim first segment prior to perturbation
    S.GUI.PostPerturbDurMultiplier = 1.5; % scaling factor for post perturbation stimulus (postperturb = preperturb * PostPerturbDurMultiplier)    
    S.GUI.MinISIPerturb_ms = 100; % min time in ms for perturbation range from grating
    S.GUI.PreVisStimDelay_s = 0; % How long the mouse must poke in the center to activate the goal port
    S.GUI.PreGoCueDelay_s = 0;
    S.GUIPanels.VisStim = {'VisStimEnable', 'GratingDur_s', 'ISIOrig_s', 'ExtraStimDurPostRew_Naive_s', 'NumISIOrigRep', 'PostPerturbDurMultiplier', 'MinISIPerturb_ms', 'PreVisStimDelay_s', 'PreGoCueDelay_s'}; 
 
    %% contingency and bias params
    S.GUI.Short_Fast_ISIChoice = 1;   % set short ISI association to left or right side
                                % 1 - 'Left', 2 - 'Right'
    S.GUIMeta.Short_Fast_ISIChoice.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
    S.GUIMeta.Short_Fast_ISIChoice.String = {'Left', 'Right'};
    S.GUI.ShortISIFraction = 0.5;   % set fraction of trials that are short ISI (long ISI fraction = (1 - short))                                
    S.GUI.ManualSideSelect = 0;   % override to disable/enable manual selection of left/right for next trial
    S.GUIMeta.ManualSideSelect.Style = 'checkbox';
    S.GUI.ManualSide = 1;   % manual selection of left/right for next trial
    S.GUIMeta.ManualSide.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
    S.GUIMeta.ManualSide.String = {'Left', 'Right'};
    S.GUIPanels.Contingency_Bias = {'Short_Fast_ISIChoice', 'ShortISIFraction', 'ManualSideSelect', 'ManualSide'};

    %% reward params
    S.GUI.LeftValveTime_s = 0.25;
    S.GUI.RightValveTime_s = 0.25;
    S.GUI.CenterValveTime_s = 0.10;
    S.GUI.WindowRewardGrabDuration_Naive_s = 10;  % naive mouse has up to x seconds to grab reward    
    S.GUI.RewardDelay_s = 0; % How long the mouse must wait in the goal port for reward to be delivered
    % S.GUI.EnableCenterLick_Trained = 1; % override to disable/enable center lick for well trained
    S.GUIMeta.EnableCenterLick_Trained.Style = 'checkbox';
    S.GUI.WindCenterLick_s = 2;   
    S.GUIPanels.Reward = {'LeftValveTime_s', 'RightValveTime_s', 'CenterValveTime_s', 'WindowRewardGrabDuration_Naive_s', 'RewardDelay_s', 'WindCenterLick_s'};

    %% punish params
    S.GUI.IncorrectSoundVolume_percent = 0.15;  % volume control
    S.GUI.PunishSoundDuration_s = 1; % Seconds to wait on errors before next trial can start
    S.GUI.IncorrectSound = 1; % if 1, plays a white noise pulse on error. if 0, no sound is played.
    S.GUIMeta.IncorrectSound.Style = 'checkbox';
    S.GUIPanels.Punish = {'IncorrectSoundVolume_percent', 'PunishSoundDuration_s', 'IncorrectSound'}; 

    %% choice params    
    S.GUI.ChoiceWindow_s = 5; % How long after go cue until the mouse must make a choice
    S.GUI.ConfirmLickInterval_s = 0.2; % min interval until choice can be confirmed    
    S.GUI.ChoiceConfirmWindow_s = 5; % time during which correct choice can be confirmed    
    S.GUIPanels.Choice = {'ChoiceWindow_s', 'ConfirmLickInterval_s', 'ChoiceConfirmWindow_s'};

    %% temp debugging params
    % S.GUI.WindowRewardGrabDuration_Naive_s = 5;  % naive mouse has up to x seconds to grab reward
    % S.GUI.ChoiceWindow_s = 2; % How long after go cue until the mouse must make a choice
    % S.GUI.NumEasyWarmupTrials = 0;
    
    % S.GUI.ITImin_s = 0;    % Minimum ITI (in seconds)
    % S.GUI.ITImax_s = 0;    % Maximum ITI (in seconds)
end

%% Display any needed informtion at start of session
disp('Running Session Setup');

%% Define trials
% set max number of trials
MaxTrials = 1000;

% get uniform distribution of 2 trial types
TrialTypes = ceil(rand(1,MaxTrials)*2); 

% adjust warmup trials to have no more than 'max' number of consecutive
% same-side trials
TrialTypes = AdjustMaxConsecutiveSameSideWarmupTrials(TrialTypes, S.GUI.NumEasyWarmupTrials);

% in case manual mode is enabled by hard-coded default, set first trial
% type accordingly
% if S.GUI.ManualSideSelect
%     switch S.GUI.ManualSide
%         case 1 % select left trial
%             % TrialTypes = ones(1, MaxTrials);
%             TrialTypes(1) = 1;
%         case 2 % select right trial
%             % TrialTypes = 2 * ones(1, MaxTrials);
%             TrialTypes(1) = 2;
%     end
% end
   
BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.

%% Initialize plots
% Side Outcome Plot
BpodSystem.ProtocolFigures.SideOutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.SideOutcomePlot = axes('Position', [.075 .35 .89 .55]);
SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'init',2-TrialTypes);
%TotalRewardDisplay('init'); % Total Reward display (online display of the total amount of liquid reward earned)
%BpodNotebook('init');
BpodParameterGUI('init', S); % Initialize parameter GUI plugin

useStateTiming = true;  % Initialize state timing plot
if ~verLessThan('matlab','9.5') % StateTiming plot requires MATLAB r2018b or newer
    useStateTiming = true;
    StateTiming();
end

%% initial distribution of difficulty levels across trials
% if (S.GUI.TrainingLevel == 1)
%     S.GUI.PercentTrialsEasy = 100;
%     S.GUI.PercentTrialsMedium = 0;
%     S.GUI.PercentTrialsHard = 0;
% 
% elseif (S.GUI.TrainingLevel == 2)
%     S.GUI.PercentTrialsEasy = 50;
%     S.GUI.PercentTrialsMedium = 30;
%     S.GUI.PercentTrialsHard = 20;
% end

%% Define stimuli and send to analog module
%SF = 192000; % Use max supported sampling rate samples/sec
SF = 44100; % Use lower sampling rate (samples/sec) to allow for longer duration audio file (max length limited by HiFi)
H.SamplingRate = SF;
Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset
%H.AMenvelope = Envelope;

InitCueSound = GenerateSineWave(SF, S.GUI.InitCueFreq_Hz, S.GUI.InitCueDuration_s)*S.GUI.InitCueVolume_percent; % Sampling freq (hz), Sine frequency (hz), duration (s)
InitCueSound = ApplySoundEnvelope(InitCueSound, Envelope);

GoCueSound = GenerateSineWave(SF, S.GUI.GoCueFreq_Hz, S.GUI.GoCueDuration_s)*S.GUI.GoCueVolume_percent; % Sampling freq (hz), Sine frequency (hz), duration (s)
GoCueSound = ApplySoundEnvelope(GoCueSound, Envelope);

IncorrectSound = GenerateWhiteNoise(SF, S.GUI.PunishSoundDuration_s, 1, 1)*S.GUI.IncorrectSoundVolume_percent; % white noise punish sound
IncorrectSound = ApplySoundEnvelope(IncorrectSound, Envelope);

% generate audio stim same duration as vis gratings
AudioStimSound = GenerateSineWave(SF, S.GUI.AudioStimFreq_Hz, S.GUI.GratingDur_s)*S.GUI.AudioStimVolume_percent; % Sampling freq (hz), Sine frequency (hz), duration (s)
AudioStimSound = ApplySoundEnvelope(AudioStimSound, Envelope);

H.DigitalAttenuation_dB = -35; % Set a comfortable listening level for most headphones (useful during protocol dev).
%H.DigitalAttenuation_dB = S.GUI.DigitalAttenuation_dB; % Set a comfortable listening level for most headphones (useful during protocol dev).
H.load(1, InitCueSound);
H.load(2, GoCueSound);
H.load(3, IncorrectSound);
%H.load(4, AudioStimSound);

% to test these sounds, can use these to play them
%H.Port.write(['P' 0], 'uint8');
%H.Port.write(['P' 1], 'uint8');
%H.Port.write(['P' 2], 'uint8');

% manually apply envelope to audio stim since it will be replicated to generate
% longer audio (needs envelope for the base waveform)
% AudioStimSoundEnvelope = Envelope;
% BackOfTheEnvelope = fliplr(AudioStimSoundEnvelope);
% IdxsBetweenTheEnvelope = length(AudioStimSound) - 2 * length(BackOfTheEnvelope);
% FullEnvelope = [AudioStimSoundEnvelope ones(1, IdxsBetweenTheEnvelope) BackOfTheEnvelope];
% AudioStimSound = AudioStimSound .* FullEnvelope;

% Remember values of sound frequencies & durations, so a new one only gets uploaded if it was changed
LastInitCueFrequency = S.GUI.InitCueFreq_Hz; 
LastGoCueFrequency = S.GUI.GoCueFreq_Hz;
LastAudioStimFrequency = S.GUI.AudioStimFreq_Hz;

LastInitCueDuration = S.GUI.InitCueDuration_s;
LastGoCueDuration = S.GUI.GoCueDuration_s;
LastPunishSoundDuration = S.GUI.PunishSoundDuration_s;

LastInitCueVolume = S.GUI.InitCueVolume_percent;
LastGoCueVolume = S.GUI.GoCueVolume_percent;
LastAudioStimVolume = S.GUI.AudioStimVolume_percent;
LastIncorrectSoundVolume = S.GUI.IncorrectSoundVolume_percent;

%% Setup video
if isfield(BpodSystem.PluginObjects, 'V') % Clear previous instances of the video server
    BpodSystem.PluginObjects.V = [];
end
MonitorID = 2;
BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(MonitorID, 0, [0 0], [180 180], 0); % Assumes second monitor is screen #2. Sync patch = 180x180 pixels

%BpodSystem.PluginObjects.V.SyncPatchIntensity = 140;
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

spatialFreq = .005; % Spatial frequency of grating in cycles per pixel
% spatialFreq = .32; % Spatial frequency of grating in cycles per pixel
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

% S.GUI.GratingDur_s = 2.00; % Duration of grating stimulus in seconds
% S.GUI.ISIOrig_s = 1.0; % Duration of *fixed* gray screen stimulus in seconds

FramesPerSecond = BpodSystem.PluginObjects.V.DetectedFrameRate;

GratingDuration = S.GUI.GratingDur_s; % set duration of grating to stimulus interval
GrayFixedDuration = S.GUI.ISIOrig_s; % set duration of gray screen to inter stimulus interval

% GratingFrames = FramesPerSecond * GratingDuration;
% GrayFixedFrames = FramesPerSecond * GrayFixedDuration;

% GratingDuration = 0.09;
% GrayFixedDuration = 0.11;

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

    % GrayPerturbFrames = convergent(FramesPerSecond * GrayPerturbDuration);
    % if (mod(GrayPerturbFrames, 2) ~= 0)
    %     GrayPerturbFrames = GrayPerturbFrames + 1; % round up to nearest even integer
    % end

 
VideoGrating = repmat(sinGrating, 1, 1, 2); % 2 frames to get sync signal encoded
VideoGrayFixed = repmat(gray, 1, 1, 2); % 2 frames to get sync signal encoded

BpodSystem.PluginObjects.V.loadVideo(1, VideoGrating);
BpodSystem.PluginObjects.V.loadVideo(2, VideoGrayFixed);

% compose grating video
GratingFrame_SyncW = BpodSystem.PluginObjects.V.Videos{1}.Data(1);
GratingFrame_SyncBlk = BpodSystem.PluginObjects.V.Videos{1}.Data(2);
GratingBlank = BpodSystem.PluginObjects.V.Videos{1}.Data(3);

%GratingPattern = [GratingFrame_SyncW GratingFrame_SyncBlk];
GratingPattern = [GratingFrame_SyncW GratingFrame_SyncW];

%GratingVideo = [repmat(GratingPattern, 1, GratingFrames/2) GratingBlank];
GratingVideo = [repmat(GratingPattern, 1, GratingFrames/2)];

BpodSystem.PluginObjects.V.Videos{3} = struct;
BpodSystem.PluginObjects.V.Videos{3}.nFrames = GratingFrames + 1; % + 1 for final frame
BpodSystem.PluginObjects.V.Videos{3}.Data = [GratingVideo GratingBlank];

% debug to check individual frames by loading them as a video to play on
% vis stim monitor
% BpodSystem.PluginObjects.V.Videos{6} = struct;
% BpodSystem.PluginObjects.V.Videos{6}.nFrames = 1; % + 1 for final frame
% BpodSystem.PluginObjects.V.Videos{6}.Data = GratingFrame_SyncW;
% BpodSystem.PluginObjects.V.play(6);
% BpodSystem.PluginObjects.V.stop;

% compose gray video, fixed ISI
GrayFrame_SyncW = BpodSystem.PluginObjects.V.Videos{2}.Data(1);
GrayFrame_SyncBlk = BpodSystem.PluginObjects.V.Videos{2}.Data(2);
GrayBlank = BpodSystem.PluginObjects.V.Videos{2}.Data(3);

%GrayPattern = [GrayFrame_SyncW GrayFrame_SyncBlk];
GrayPattern = [GrayFrame_SyncBlk GrayFrame_SyncBlk];

%GrayVideo = [repmat(GrayPattern, 1, GrayFixedFrames/2) GrayBlank];
GrayVideo = [repmat(GrayPattern, 1, GrayFixedFrames/2)];

% update durations based on number of frames generated
GratingDur = length(GratingVideo) * (1/FramesPerSecond);
GrayDur = length(GrayVideo) * (1/FramesPerSecond);

BpodSystem.PluginObjects.V.Videos{4} = struct;
BpodSystem.PluginObjects.V.Videos{4}.nFrames = GrayFixedFrames + 1; % + 1 for final frame
BpodSystem.PluginObjects.V.Videos{4}.Data = [GrayVideo GrayBlank];

VideoPrePerturbPattern = [GratingVideo GrayVideo]; % base video pattern for initial segment repetitions of grating->gray

% try to get Frame2TTL BNC sync to be low and not miss first frame of vis
% stim later
GrayInitBNCSync = [repmat(GrayFrame_SyncW, 1, 120) GrayFrame_SyncBlk];
BpodSystem.PluginObjects.V.Videos{6} = struct;
BpodSystem.PluginObjects.V.Videos{6}.nFrames = 121; % + 1 for final frame
BpodSystem.PluginObjects.V.Videos{6}.Data = GrayInitBNCSync;

%NumISIOrigRep = S.GUI.NumISIOrigRep;

VideoData = [repmat(VideoPrePerturbPattern, 1, S.GUI.NumISIOrigRep) GratingVideo]; % add one more grating video after initial repetitions

PrePerturbDur = (GratingDuration + GrayFixedDuration) * S.GUI.NumISIOrigRep + GratingDuration;

if ShowDebugOutput
    disp(['PrePerturbDur: ', num2str(PrePerturbDur)]);
end

% %NumISIOrigRep = 5;
% for BaseSegments = 1:NumISIOrigRep
%     %BaseVideo = [BaseVideo repmat(GratingPattern, 1, GratingFrames/2) repmat(GrayPattern, 1, GrayFixedFrames/2)];
%     BaseVideo = [BaseVideo GratingVideo GrayVideo];
% end
% % BaseVideo = [BaseVideo repmat(GratingPattern, 1, GratingFrames/2)];
% BaseVideo = [BaseVideo GratingVideo];
% 
% VideoData = BaseVideo;  % store initial segment in 3d video matrix

LastGratingDuration = S.GUI.GratingDur_s; % Remember value of stim dur so that we only regenerate the grating video if parameter has changed
LastGrayFixedDuration = S.GUI.ISIOrig_s; % Remember value of pre-perturb gray dur so that we only regenerate the pre-perturb gray video if parameter has changed
LastNumISIOrigRep = S.GUI.NumISIOrigRep;  % Remember value of initial segment repetitions so that we only regenerate the initial segment if parameter changed

% PerturbDurMin = 30;% Time in ms % Oddball min and max interval duration; uniform distribution.
% PerturbDurMax = 3000; % Time in ms
% RandomPerturbationDur = unifrnd(PerturbDurMin, PerturbDurMax, 1, 1)/1000; % get random duration, convert to seconds
% disp('RandomPerturbationDur:');
% disp(RandomPerturbationDur);
% GrayPerturbDuration = GrayFixedDuration + RandomPerturbationDur; 
% GrayPerturbFrames = convergent(FramesPerSecond * GrayPerturbDuration);
% if (mod(GrayPerturbFrames, 2) ~= 0)
%     GrayPerturbFrames = GrayPerturbFrames + 1; % round up to nearest even integer
% end
% GrayPerturbVideo = [repmat(GrayPattern, 1, GrayPerturbFrames/2)];
% 
% FullVideo = [VideoData GrayPerturbVideo GratingVideo GrayPerturbVideo GratingVideo GrayBlank];
% FullVideoFrames = S.GUI.NumISIOrigRep * (GratingFrames + GrayFixedFrames) + 3 * GratingFrames + 2 * GrayPerturbFrames + 1; % + 1 for final frame
% VisStimDuration = FullVideoFrames / FramesPerSecond;
% 
% BpodSystem.PluginObjects.V.Videos{5} = struct;
% BpodSystem.PluginObjects.V.Videos{5}.nFrames = FullVideoFrames; 
% BpodSystem.PluginObjects.V.Videos{5}.Data = FullVideo;

% BpodSystem.PluginObjects.V.loadVideo(1, MyVideoL);
% BpodSystem.PluginObjects.V.loadVideo(2, MyVideoR);
BpodSystem.PluginObjects.V.TimerMode = 0;
pause(1.0); % matlab seems to require a pause here before clearing screen with play(0), 
            % otherwise can get stuck on Psychtoolbox splash screen
            % might need longer delay if purple image hangs on window open
BpodSystem.PluginObjects.V.play(0);
BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_PlayVideo';

BpodSystem.PluginObjects.V.play(6);
BpodSystem.PluginObjects.V.TimerMode = 2;
% wait for parameter update and confirm before beginning trial loop
input('Set parameters and press enter to continue >', 's'); 
S = BpodParameterGUI('sync', S);

%% check difficulty options and ensure correct setting prior to beginning first trial
% NumEasy = floor(S.GUI.PercentTrialsEasy*MaxTrials);
% NumMedium = floor(S.GUI.PercentTrialsMedium*MaxTrials);
% NumHard = floor(S.GUI.PercentTrialsHard*MaxTrials);
% 
% DifficultySum = NumEasy + NumMedium + NumHard;

% define discrete values in distribution
%DifficultyLevels = [1, 2, 3];  % 1 - Easy, 2 - Medium, 3 - Hard
DifficultyLevels = [1, 2, 3, 4];  % 1 - Easy, 2 - MediumEasy, 3 - MediumHard, 4 - Hard

WarmupTrialsCounter = S.GUI.NumEasyWarmupTrials;
LastNumEasyWarmupTrials = S.GUI.NumEasyWarmupTrials; % store GUI value to determine if user has changed this param to reset counter

% check that difficulty level distribution percentages sum to 100%
%PercentDifficultySum = S.GUI.PercentTrialsEasy + S.GUI.PercentTrialsMedium + S.GUI.PercentTrialsHard;
PercentDifficultySum = S.GUI.PercentTrialsEasy + S.GUI.PercentTrialsMediumEasy + S.GUI.PercentTrialsMediumHard + S.GUI.PercentTrialsHard;

if ShowDebugOutput
    disp(['PercentDifficultySum:', num2str(PercentDifficultySum)]);
end

% require operator to set difficulty level distribution percentages that sum to 100%
%while PercentDifficultySum ~= 1
DifficultyPercentageSumTolerance = 0.1;
while (PercentDifficultySum < (100-DifficultyPercentageSumTolerance)) || (PercentDifficultySum > (100+DifficultyPercentageSumTolerance))
    input('Sum of difficulty level percentages must be 100%, after setting press enter to continue >', 's'); 
    S = BpodParameterGUI('sync', S);
    %PercentDifficultySum = S.GUI.PercentTrialsEasy + S.GUI.PercentTrialsMedium + S.GUI.PercentTrialsHard;
    PercentDifficultySum = S.GUI.PercentTrialsEasy + S.GUI.PercentTrialsMediumEasy + S.GUI.PercentTrialsMediumHard + S.GUI.PercentTrialsHard;
    disp(['PercentDifficultySum:', num2str(PercentDifficultySum)]);
end

% NumEasy = floor(S.GUI.PercentTrialsEasy/100*MaxTrials);
% NumMedium = floor(S.GUI.PercentTrialsMedium/100*MaxTrials);
% NumHard = floor(S.GUI.PercentTrialsHard/100*MaxTrials);
% 
% DifficultySum = NumEasy + NumMedium + NumHard;
% 
% TrialsWithUnassignedDifficulty = MaxTrials - DifficultySum;
% NumEasy = NumEasy + TrialsWithUnassignedDifficulty;
% 
% DifficultySum = NumEasy + NumMedium + NumHard;

% make sure these correspond to the correct value chosen to symbolize
% difficulty level
% DiffTypeEasy = 1;
% DiffTypeMedium = 2;
% DiffTypeHard = 3;

% EasyArr = repmat(DiffTypeEasy, 1, NumEasy);
% MediumArr = repmat(DiffTypeMedium, 1, NumMedium);
% HardArr = repmat(DiffTypeHard, 1, NumHard);
% EasyArr = ones(1, NumEasy) * DiffTypeEasy;
% MediumArr = ones(1, NumMedium) * DiffTypeMedium;
% HardArr = ones(1, NumHard) * DiffTypeHard;
% 
% DifficultyArr = [EasyArr MediumArr HardArr];
% RandDifficultyArr = DifficultyArr(randperm(length(DifficultyArr)));

%% Main trial loop
for currentTrial = 1:MaxTrials
    ExperimenterTrialInfo.TrialNumber = currentTrial;   % capture variable states as field/value struct for experimenter info

    %% sync trial-specific parameters from GUI
    S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin    

    %% account for contingency settings

    switch S.GUI.ManualSideSelect
        case 0 % manual selection disabled, use ShortISIFraction if it is ~= 0.5
            if (S.GUI.ShortISIFraction ~= 0.5)
                switch S.GUI.Short_Fast_ISIChoice
                    case 1  % left is short ISI
                        % define discrete side values in distribution
                        Sides = [1, 2];                    
                    case 2  % right is short ISI
                        % define discrete side values in distribution
                        Sides = [2, 1]; 
                end
                % assigned probability weights for sampling from distribution
                SideProbabilities = [S.GUI.ShortISIFraction, 1-S.GUI.ShortISIFraction];
            
                % define discrete case values in distribution
                %Sides = [1, 2];
                % assigned probability weights for sampling from distribution
                %SideProbabilities = [S.GUI.ShortISIFraction, 1-S.GUI.ShortISIFraction];
                
                % check that difficulty probability weights sum to 1.0
                % maybe change this to message using disp() so it doesn't look like program
                % error/crash
                % AssertCondition = (sum(DifficultyProbabilities) == 1);
                % AssertErrMsg = ['Sum of difficulty probability weights must equal 1.0'];
                % assert(AssertCondition, AssertErrMsg);
                                
                cp = [0, cumsum(SideProbabilities)]; % cumulative probability -> use as interval to pick Left or Right

                r = rand; % get random scalar drawn from the uniform distribution in the interval (0,1).
                ind = find(r>cp, 1, 'last');  % get discrete index (1 or 2 for left or right)
                TrialTypes(currentTrial) = Sides(ind); % get discrete value at the randomly (according to probability weights) selected index, 
                                                       % in this case of 1 = Left, 2 = Right, it will be the same as the index.  This step is 
                                                       % here in case more or fewer cases are added in the future, this gets used as example for drawing 
                                                       % from weighted distribution later, or any other unforseen reason
            end
        case 1 % manual selection enabled, use selected side, project future trials as if current setting stays selected
            switch S.GUI.ManualSide
                case 1 % set trial as 'Left'
                    TrialTypes(currentTrial) = 1;
                case 2 % set trial as 'Right'
                    TrialTypes(currentTrial) = 2;
            end
    end

    %% update trial-specific valve times according to set reward amount
    % maybe move this section down
    %R = GetValveTimes(S.GUI.RewardAmount, [1 3]); LeftValveTime = R(1); RightValveTime = R(2); % Update reward amounts
    
    % LeftValveTime = 0.02; CenterValveTime = 0.02; RightValveTime = 0.02;
    % LeftValveTime = 1; CenterValveTime = 0.5; RightValveTime = 1;
    % LeftValveTime = 0.25; CenterValveTime = 0.05; RightValveTime = 0.25;
    
    LeftValveTime = S.GUI.LeftValveTime_s;
    RightValveTime = S.GUI.RightValveTime_s;
    CenterValveTime = S.GUI.CenterValveTime_s;

    %% set state matrix vars according to contigency

    switch TrialTypes(currentTrial) % Determine trial-specific state matrix fields
        case 1 % left side is rewarded
            CorrectLick = 'Port1In';
            CorrectPort = 'Port1';
            IncorrectLick = 'Port3In';
            IncorrectPort = 'Port3';
            %StimulusOutput = {'SoftCode', 3};
            %RewardPortOut = 'Port1Out'; 
            ValveTime = LeftValveTime; 
            Valve = 'Valve1';
            %CorrectWithdrawalEvent = 'Port1Out';
            %VisualGratingDuration = S.GUI.LDuration; % length of time to play visual stimulus when Left TrialType
    
            %ExperimenterTrialInfo.RewardedSide = 'Left';   % capture variable states as field/value struct for experimenter info
        case 2  % right side is rewarded 
            CorrectLick = 'Port3In';
            CorrectPort = 'Port3';
            IncorrectLick = 'Port1In';
            IncorrectPort = 'Port1';
            %StimulusOutput = {'SoftCode', 4};
            %RewardPortOut = 'Port3Out'; 
            ValveTime = RightValveTime; 
            Valve = 'Valve3';
            %CorrectWithdrawalEvent = 'Port3Out';
            %VisualGratingDuration = S.GUI.RDuration; % length of time to play visual stimulus when Right TrialType
    
            %ExperimenterTrialInfo.RewardedSide = 'Right';   % capture variable states as field/value struct for experimenter info    
    end

    %% get difficulty level for this trial
    
    % check if user has changed number of warmup trials    
    if S.GUI.NumEasyWarmupTrials ~= LastNumEasyWarmupTrials
        WarmupTrialsCounter = S.GUI.NumEasyWarmupTrials;    % update warmap trial counter to current gui param
        LastNumEasyWarmupTrials = S.GUI.NumEasyWarmupTrials;    % store currenct value to check for change again
    end

    % if warmup trial, choose easy, otherwise pick difficulty from
    % weighted probability distribution    
    if WarmupTrialsCounter > 0
        if ShowDebugOutput
            disp('warmup trial');
            disp(['remaining warmup trials (including this one): ', num2str(WarmupTrialsCounter)]);
        end

        ExperimenterTrialInfo.Warmup = true;   % capture variable states as field/value struct for experimenter info
        ExperimenterTrialInfo.WarmupTrialsRemaining = WarmupTrialsCounter;   % capture variable states as field/value struct for experimenter info

        TrialDifficulty = 1;  % set warmup trial to easy
        WarmupTrialsCounter = WarmupTrialsCounter - 1;        
    else
        % simplify this if we're not going to allow user-specified gui
        % params for distribution percentage, then don't need input
        % validation, only need to check for training level each trial then
        % pick

        % if (S.GUI.TrainingLevel == 1)
        %     PercentTrialsEasy = 100;
        %     PercentTrialsMedium = 0;
        %     PercentTrialsHard = 0;
        % 
        % elseif (S.GUI.TrainingLevel == 2)
        %     PercentTrialsEasy = 50;
        %     PercentTrialsMedium = 30;
        %     PercentTrialsHard = 20;
        % end     

        PercentTrialsEasy = S.GUI.PercentTrialsEasy;
        %PercentTrialsMedium = S.GUI.PercentTrialsMedium;
        PercentTrialsMediumEasy = S.GUI.PercentTrialsMediumEasy;
        PercentTrialsMediumHard = S.GUI.PercentTrialsMediumHard;
        PercentTrialsHard = S.GUI.PercentTrialsHard;

        % get probabilities from percentages
        % FractionEasy = S.GUI.PercentTrialsEasy/100;
        % FractionMedium = S.GUI.PercentTrialsMedium/100;
        % FractionHard = S.GUI.PercentTrialsHard/100;
        FractionEasy = PercentTrialsEasy/100;
        %FractionMedium = PercentTrialsMedium/100;
        FractionMediumEasy = PercentTrialsMediumEasy/100;
        FractionMediumHard = PercentTrialsMediumHard/100;
        FractionHard = PercentTrialsHard/100;
        
        % assigned probability weights for sampling from distribution
        %DifficultyProbabilities = [FractionEasy, FractionMedium, FractionHard];
        DifficultyProbabilities = [FractionEasy, FractionMediumEasy, FractionMediumHard, FractionHard];
        
        % check that difficulty probability weights sum to 1.0
        % maybe change this to message using disp() so it doesn't look like program
        % error/crash
        ProbabilitySum1 = (sum(DifficultyProbabilities) == 1);
        
        % don't move forward with trials until the entered easy/medium/hard
        % fractions sum to 100%
        while ~ProbabilitySum1
            % AssertErrMsg = ['Sum of difficulty percentages must equal 100'];
            % assert(ProbabilitySum1, AssertErrMsg);
            disp('Sum of difficulty percentages must equal 100');
            input('Press enter after setting difficulty percentage sum equal to 100 to continue >', 's'); 
            S = BpodParameterGUI('sync', S);
    
            % update difficulty percentages from gui params
            PercentTrialsEasy = S.GUI.PercentTrialsEasy;
            %PercentTrialsMedium = S.GUI.PercentTrialsMedium;
            PercentTrialsMediumEasy = S.GUI.PercentTrialsMediumEasy;
            PercentTrialsMediumHard = S.GUI.PercentTrialsMediumHard;
            PercentTrialsHard = S.GUI.PercentTrialsHard;

            % get probabilities from percentages
            % FractionEasy = S.GUI.PercentTrialsEasy/100;
            % FractionMedium = S.GUI.PercentTrialsMedium/100;
            % FractionHard = S.GUI.PercentTrialsHard/100;
            FractionEasy = PercentTrialsEasy/100;
            %FractionMedium = PercentTrialsMedium/100;
            FractionMediumEasy = PercentTrialsMediumEasy/100;
            FractionMediumHard = PercentTrialsMediumHard/100;
            FractionHard = PercentTrialsHard/100;
            
            % assigned probability weights for sampling from distribution
            %DifficultyProbabilities = [FractionEasy, FractionMedium, FractionHard];
            DifficultyProbabilities = [FractionEasy, FractionMediumEasy, FractionMediumHard, FractionHard];
            
            % check that difficulty probability weights sum to 1.0
            % maybe change this to message using disp() so it doesn't look like program
            % error/crash
            ProbabilitySum1 = (sum(DifficultyProbabilities) == 1);
        end
    
        % if we've reached here, then the weights sum to a probability of 1.0,
        % now draw random sample according to probability weights
        cp = [0, cumsum(DifficultyProbabilities)]; % cumulative probability -> use as interval to pick Easy OR Medium OR Hard (one occurs every draw)
        r = rand; % get random scalar drawn from the uniform distribution in the interval (0,1).
        ind = find(r>cp, 1, 'last');  % get discrete index (1, 2, or 3 for Easy, Medium, or Hard in this case)
        TrialDifficulty = DifficultyLevels(ind); % get discrete value at the randomly (according to probability weights) selected index,
                                        % in this case of 1 = Easy, 2 = Medium, 3 = Hard it will be the same as the index.  
                                        % This step is here in case more or fewer difficulty levels are added in the future, 
                                        % this gets used as example for drawing from weighted distribution later, or any other unforseen reason
        
        ExperimenterTrialInfo.Warmup = false;   % capture variable states as field/value struct for experimenter info
        ExperimenterTrialInfo.WarmupTrialsRemaining = 0;   % capture variable states as field/value struct for experimenter info
    end    

    switch TrialDifficulty
        case 1
            ExperimenterTrialInfo.Difficulty = 'Easy';   % capture variable states as field/value struct for experimenter info
        case 2
            ExperimenterTrialInfo.Difficulty = 'Medium-Easy';   % capture variable states as field/value struct for experimenter info
        case 3
            ExperimenterTrialInfo.Difficulty = 'Medium-Hard';   % capture variable states as field/value struct for experimenter info
        case 4
            ExperimenterTrialInfo.Difficulty = 'Hard';   % capture variable states as field/value struct for experimenter info
    end

    % print difficulty drawn from prob dist
    if ShowDebugOutput
        switch TrialDifficulty
            case 1
                disp('Easy Picked');
            case 2
                disp('Medium-Easy Picked');
            case 3
                disp('Medium-Hard Picked');                
            case 4
                disp('Hard Picked');
        end
    end

    %% Draw trial-specific ITI from exponential distribution
    % ITI exponential distribution Parameters
    % S.GUI.ITImin_s = 1;    % Minimum ITI (in seconds)
    % S.GUI.ITImax_s = 5;    % Maximum ITI (in seconds)
    % S.GUI.ITIlambda = 0.3;  % ITIlambda parameter of the exponential distribution
    %S.GUI.ITISetExplicit = 0;  % flag to manually set ITI instead of drawing from distribution
    
    % Generate a random value from the exponential distribution
    ITI = -log(rand) / S.GUI.ITIlambda;
    
    % Check if the generated ITI is within the desired range
    while ITI < S.GUI.ITImin_s || ITI > S.GUI.ITImax_s
        ITI = -log(rand) / S.GUI.ITIlambda;
    end

    if ForceITIZero
        ITI = 0;
    end

    ExperimenterTrialInfo.ITI = ITI;   % capture variable states as field/value struct for experimenter info

    %ITI = 0;

    if ShowDebugOutput
        % Display the generated ITI
        disp(['Generated ITI:', num2str(ITI)]);
    end
    %% construct preperturb vis stim videos and audio stim base for grating and gray if duration parameters changed
    
    if S.GUI.GratingDur_s ~= LastGratingDuration
        if ShowDebugOutput
            disp(['changed grating dur from: ', num2str(LastGratingDuration), ' to: ', num2str(S.GUI.GratingDur_s)]);
        end
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
        
        % BpodSystem.PluginObjects.V.Videos{3} = struct;
        % BpodSystem.PluginObjects.V.Videos{3}.nFrames = GratingFrames + 1; % + 1 for final frame
        % BpodSystem.PluginObjects.V.Videos{3}.Data = [GratingVideo GratingBlank];   
        %GratingDur = length(GratingVideo) * (1/FramesPerSecond);
        %AudioStimSound = GenerateSineWave(SF, S.GUI.AudioStimFreq_Hz, GratingDur); % Sampling freq (hz), Sine frequency (hz), duration (s)
        %H.load(4, AudioStimSound);

        % update durations based on number of frames generated
        GratingDur = length(GratingVideo) * (1/FramesPerSecond);
        %GrayDur = length(GrayVideo) * (1/FramesPerSecond);
    end
    if S.GUI.ISIOrig_s ~= LastGrayFixedDuration
        if ShowDebugOutput
            disp(['changed gray dur from: ', num2str(LastGrayFixedDuration), ' to: ', num2str(S.GUI.ISIOrig_s)]);
        end
        GrayFixedDuration = S.GUI.ISIOrig_s; % set duration of gray screen to inter stimulus interval

        GrayFixedFrames = convergent(FramesPerSecond * GrayFixedDuration);
        if (mod(GrayFixedFrames, 2) ~= 0)
            GrayFixedFrames = GrayFixedFrames + 1; % round up to nearest even integer
        end
       
        % compose gray video, fixed ISI
        GrayVideo = [repmat(GrayPattern, 1, GrayFixedFrames/2)];
               
        % BpodSystem.PluginObjects.V.Videos{4} = struct;
        % BpodSystem.PluginObjects.V.Videos{4}.nFrames = GrayFixedFrames + 1; % + 1 for final frame
        % BpodSystem.PluginObjects.V.Videos{4}.Data = [GrayVideo GrayBlank];    

        % update durations based on number of frames generated
        %GratingDur = length(GratingVideo) * (1/FramesPerSecond);
        GrayDur = length(GrayVideo) * (1/FramesPerSecond);        
    end
    %% Generate Audio Stim again if freq changed or if grating dur changed
    if S.GUI.AudioStimFreq_Hz ~= LastAudioStimFrequency
        % generate audio stim of 15s (longer than vis stim to go cue would
        % be) then turn sound off at beginning of next state after vis stim
        %AudioStimSound = GenerateSineWave(SF, S.GUI.AudioStimFreq_Hz, GratingDur); % Sampling freq (hz), Sine frequency (hz), duration (s)
        %H.load(4, [AudioStimSound;AudioStimSound]);
        %LastAudioStimFrequency = S.GUI.AudioStimFreq_Hz;
    end 
    %% update video& audio and change tracking variables for adio and vis stim
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
    % if stim or isi dur changed, initial video pattern has to be
    % reconstructed from parts above
    if (S.GUI.GratingDur_s ~= LastGratingDuration) || (S.GUI.ISIOrig_s ~= LastGrayFixedDuration) || (S.GUI.NumISIOrigRep ~= LastNumISIOrigRep)
        if ShowDebugOutput
            disp('initial vid reconstructed')
        end
        VideoPrePerturbPattern = [GratingVideo GrayVideo]; % base video pattern for initial segment repetitions of grating->gray
        VideoData = [repmat(VideoPrePerturbPattern, 1, S.GUI.NumISIOrigRep) GratingVideo]; % construct initial video segment, add one more grating video after initial repetitions
        LastGratingDuration = S.GUI.GratingDur_s; % Remember value of stim dur so that we only regenerate the grating video if parameter has changed
        LastGrayFixedDuration = S.GUI.ISIOrig_s; % Remember value of stim dur so that we only regenerate the grating video if parameter has changed
        LastNumISIOrigRep = S.GUI.NumISIOrigRep;  % Remember value of initial segment repetitions so that we only regenerate the initial segment if parameter changed
    end
       

    % % reconstruct base if parameter was changed
    % if S.GUI.NumISIOrigRep ~= LastNumISIOrigRep
    %     VideoData = [repmat(VideoPrePerturbPattern, 1, S.GUI.NumISIOrigRep) GratingVideo]; % construct initial video segment, add one more grating video after initial repetitions
    % 
    %     LastNumISIOrigRep = S.GUI.NumISIOrigRep;  % Remember value of initial segment repetitions so that we only regenerate the initial segment if parameter changed
    % end

    %% set vis stim perturbation ISI duration according to trial-specific difficulty level
    % draw perturbation interval from uniform distribution in range according to difficulty
    
    % %RandomDurationRangeMin = 70;% min time in ms for perturbation uniform distribution. - ORIGINAL
    % RandomDurationRangeMin = 100;% min time in ms for perturbation uniform distribution.
    % %PerturbDurMax = S.GUI.ISIOrig_s * 1000 - PerturbDurMin; % Time in ms, need to have random duration be at most the ISI, 
    % RandomDurationRangeMax = S.GUI.ISIOrig_s * 1000; % Time in ms, need to have random duration be at most the ISI, 
    
    % S.GUI.EasyMinPercent = 2/3*100;
    % S.GUI.EasyMaxPercent = 1*100;
    % S.GUI.MediumMinPercent = 1/3*100;
    % S.GUI.MediumMaxPercent = 2/3*100;
    % S.GUI.HardMinPercent = 0*100;
    % S.GUI.HardMaxPercent = 1/3*100;
    % EasyMinPercent = S.GUI.EasyMinPercent/100;
    % EasyMaxPercent = S.GUI.EasyMaxPercent/100;
    % MediumMinPercent = S.GUI.MediumMinPercent/100;
    % MediumMaxPercent = S.GUI.MediumMaxPercent/100;
    % HardMinPercent = S.GUI.HardMinPercent/100;
    % HardMaxPercent = S.GUI.HardMaxPercent/100;    
    
    % get current percentage boundaries from gui params
    % EasyMinPercent = 2/3;
    % EasyMaxPercent = 1;
    % MediumMinPercent = 1/3;
    % MediumMaxPercent = 2/3;
    % HardMinPercent = 0;
    % HardMaxPercent = 1/3;

    EasyMinPercent = 3/4;
    EasyMaxPercent = 1;
    % MediumMinPercent = 1/3;
    % MediumMaxPercent = 2/3;
    MediumEasyMinPercent = 1/2;
    MediumEasyMaxPercent = 3/4;
    MediumHardMinPercent = 1/4;
    MediumHardMaxPercent = 1/2;
    HardMinPercent = 0;
    HardMaxPercent = 1/4;
   
    % calculate category boundary to use as base timing reference for
    % perturbation length
    %CatBound = S.GUI.ISIOrig_s;   % updated category boundary to be ISI_perturb * 1.5
    CatBound = S.GUI.ISIOrig_s;   % updated category boundary to be ISI_perturb * 1.5

    ExperimenterTrialInfo.CategoryBoundary = CatBound;   % capture variable states as field/value struct for experimenter info

    % full range of perturbation is calculated as the category boundary time
    % minus the selected minimum time separation from the category boundary and the grating
    %PerturbDurFullRange = S.GUI.ISIOrig_s * 1000 - S.GUI.PerturbMinFromCB_ms - S.GUI.MinISIPerturb_ms;  
    
    %PerturbDurFullRange = S.GUI.ISIOrig_s * 1000 - S.GUI.MinISIPerturb_ms; 
    PerturbDurFullRange = CatBound * 1000 - S.GUI.MinISIPerturb_ms; 
                                                                                                     
    switch TrialDifficulty % Determine trial-specific visual stimulus
        case 1
            if ShowDebugOutput
                disp('Difficulty - Easy');
            end
            PerturbDurMin = EasyMinPercent*PerturbDurFullRange;
            PerturbDurMax = EasyMaxPercent*PerturbDurFullRange;
        % case 2
        %     if ShowDebugOutput
        %         disp('Difficulty - Medium');
        %     end
        %     PerturbDurMin = MediumMinPercent*PerturbDurFullRange;
        %     PerturbDurMax = MediumMaxPercent*PerturbDurFullRange;
        case 2
            if ShowDebugOutput
                disp('Difficulty - Medium-Easy');
            end
            PerturbDurMin = MediumEasyMinPercent*PerturbDurFullRange;
            PerturbDurMax = MediumEasyMaxPercent*PerturbDurFullRange;
        case 3
            if ShowDebugOutput
                disp('Difficulty - Medium-Hard');
            end
            PerturbDurMin = MediumHardMinPercent*PerturbDurFullRange;
            PerturbDurMax = MediumHardMaxPercent*PerturbDurFullRange;            
        case 4
            if ShowDebugOutput
                disp('Difficulty - Hard');
            end
            PerturbDurMin = HardMinPercent*PerturbDurFullRange;
            PerturbDurMax = HardMaxPercent*PerturbDurFullRange;
    end

    % after applying difficulty to perturbation range, shift the min and
    % max away to the selected minimum from category boundary
    % PerturbDurMin = PerturbDurMin + S.GUI.PerturbMinFromCB_ms;
    % PerturbDurMax = PerturbDurMax + S.GUI.PerturbMinFromCB_ms;

    debugDiffRanges = 1;
    if debugDiffRanges
        % EasyMin = (EasyMinPercent*PerturbDurFullRange + S.GUI.PerturbMinFromCB_ms)/1000;
        % EasyMax = (EasyMaxPercent*PerturbDurFullRange + S.GUI.PerturbMinFromCB_ms)/1000;
        % MediumMin = (MediumMinPercent*PerturbDurFullRange + S.GUI.PerturbMinFromCB_ms)/1000;
        % MediumMax = (MediumMaxPercent*PerturbDurFullRange + S.GUI.PerturbMinFromCB_ms)/1000;
        % HardMin = (HardMinPercent*PerturbDurFullRange + S.GUI.PerturbMinFromCB_ms)/1000;
        % HardMax = (HardMaxPercent*PerturbDurFullRange + S.GUI.PerturbMinFromCB_ms)/1000;
        EasyMin = (EasyMinPercent*PerturbDurFullRange)/1000;
        EasyMax = (EasyMaxPercent*PerturbDurFullRange)/1000;
        % MediumMin = (MediumMinPercent*PerturbDurFullRange)/1000;
        % MediumMax = (MediumMaxPercent*PerturbDurFullRange)/1000;
        MediumEasyMin = (MediumEasyMinPercent*PerturbDurFullRange)/1000;
        MediumEasyMax = (MediumEasyMaxPercent*PerturbDurFullRange)/1000;
        MediumHardMin = (MediumHardMinPercent*PerturbDurFullRange)/1000;
        MediumHardMax = (MediumHardMaxPercent*PerturbDurFullRange)/1000;        
        HardMin = (HardMinPercent*PerturbDurFullRange)/1000;
        HardMax = (HardMaxPercent*PerturbDurFullRange)/1000;

        if ShowDebugOutput
            disp(['Easy Range:', num2str(EasyMin),' - ', num2str(EasyMax)]);
            % disp(['Medium Range:', num2str(MediumMin),' - ', num2str(MediumMax)]);
            disp(['Medium-Easy Range:', num2str(MediumEasyMin),' - ', num2str(MediumEasyMax)]);
            disp(['Medium-Hard Range:', num2str(MediumHardMin),' - ', num2str(MediumHardMax)]);
            disp(['Hard Range:', num2str(HardMin),' - ', num2str(HardMax)]);
        end
    end
    
    if (S.GUI.TrainingLevel == 1 || S.GUI.TrainingLevel == 2)
        RandomPerturbationDur = EasyMax;
        if ShowDebugOutput
            disp(['Naive -> EasyMax:']);        
        end
    else
        RandomPerturbationDur = unifrnd(PerturbDurMin, PerturbDurMax, 1, 1)/1000; % get random duration from calculated range, convert to seconds
    end
    
    % % explicit random duration for debugging video
    %RandomPerturbationDur = 1;
    %RandomPerturbationDur = S.GUI.ISIOrig_s;    

    ExperimenterTrialInfo.DistanceFromCategoryBoundary = RandomPerturbationDur;   % capture variable states as field/value struct for experimenter info

    if ShowDebugOutput
        disp(['RandomPerturbationDur:', num2str(RandomPerturbationDur)]);        
    end

    % S.GUI.Short_Fast_ISIChoice
    % S.GUI.ShortISIFraction
    % S.GUI.ManualSideSelect
    % S.GUI.ManualSide
   
    % set short and long ISI according to contingency defined by gui params
    % maybe collapse this into smaller if conditions
    switch S.GUI.Short_Fast_ISIChoice
        case 1 % left is short ISI        
            switch TrialTypes(currentTrial) % Determine trial-specific visual stimulus duration
                case 1 % trial is left with short ISI                    
                    GrayPerturbDuration = CatBound - RandomPerturbationDur; % for left reward, ISI is subtractd from the random duration            
                case 2 % trial is right with long ISI                    
                    GrayPerturbDuration = CatBound + RandomPerturbationDur; % for right reward, ISI is added to the random duration
            end
        case 2 % right is short ISI
            switch TrialTypes(currentTrial) % Determine trial-specific visual stimulus duration
                case 1 % trial is left with long ISI                    
                    GrayPerturbDuration = CatBound + RandomPerturbationDur; % for left reward, ISI is added from the random duration            
                case 2 % trial is right with short ISI                    
                    GrayPerturbDuration = CatBound - RandomPerturbationDur; % for right reward, ISI is subtractd to the random duration
            end
    end
    
    UpdateSideOutcomePlot(TrialTypes, BpodSystem.Data, 0); % this will update the SideOutcomePlot to reflect the current trial type after accounting for any change due to anti-bias control 
  

    % % explicit gray duration for debugging video
    %GrayPerturbDuration = S.GUI.ISIOrig_s - 0.1;   
    
    if ShowDebugOutput
        disp(['GrayPerturbDuration (perturb ISI):', num2str(GrayPerturbDuration)]);        
    end

    debugGrayPerturbDuration = GrayPerturbDuration;

    %% Construct trial-specific portion of video and add it to base video
    % construct video for this trial
    
    % find number of frames for variable ISI    
    if ShowDebugOutput
        disp(['FPS:', num2str(FramesPerSecond)]);        
    end

    GrayPerturbFrames = convergent(FramesPerSecond * GrayPerturbDuration); % rounds ties to the nearest even integer

    if ShowDebugOutput
        disp(['GrayPerturbFrames: ', num2str(GrayPerturbFrames)]);
    end

    if (mod(GrayPerturbFrames, 2) ~= 0)
        GrayPerturbFrames = GrayPerturbFrames + 1; % round up to nearest even integer

        if ShowDebugOutput
            disp(['GrayPerturbFrames (Adj): ', num2str(GrayPerturbFrames)]);
        end
    end
    GrayPerturbVideo = [repmat(GrayPattern, 1, GrayPerturbFrames/2)];

    GrayPerturbDuration = GrayPerturbFrames / FramesPerSecond; % convert even rounded number of frames back into duration to calculate video duration

    if ShowDebugOutput
        disp(['GrayPerturbDuration: ', num2str(debugGrayPerturbDuration)]); % from random selection + ISI above in code, debug comparison
        disp(['GrayPerturbDuration (Adj): ', num2str(GrayPerturbDuration)]);
        disp(['RandomPerturbationDur: ', num2str(RandomPerturbationDur)]); % from random selection above in code, debug comparison
        disp(['RandomPerturbationDur (Adj): ', num2str(abs(GrayPerturbDuration - S.GUI.ISIOrig_s))]);
    end

    ExperimenterTrialInfo.ISIPerturbDuration = GrayPerturbDuration;   % capture variable states as field/value struct for experimenter info

    VideoPerturbBasePattern = [GrayPerturbVideo GratingVideo]; % perturbation video pattern for second segment repetitions of random ISI gray->grating

    if ShowDebugOutput
        disp(['length(VideoPerturbBasePattern): ', num2str(length(VideoPerturbBasePattern))]);
    end
   
    % find the nearest whole number duration of perturbation gray->grating that fits
    % within the same duration as the pre-preturbation segment
    % PerturbBasePatternDuration = GrayPerturbDuration + GratingDuration;
    % PrePerturbDuration = length(VideoData) / FramesPerSecond;
    % NumPerturbVisRep = floor(PrePerturbDuration/PerturbBasePatternDuration);

    % find the nearest whole number duration of perturbation gray->grating that fits
    % within the scaled duration of the pre-preturbation segment
    PerturbBasePatternDuration = GrayPerturbDuration + GratingDuration;
    ScaledPrePerturbDuration = (length(VideoData) / FramesPerSecond) * S.GUI.PostPerturbDurMultiplier;
    NumPerturbVisRep = floor(ScaledPrePerturbDuration/PerturbBasePatternDuration);

    if ShowDebugOutput
        disp(['PerturbBasePatternDuration: ', num2str(PerturbBasePatternDuration)]);
        disp(['ScaledPrePerturbDuration: ', num2str(ScaledPrePerturbDuration)]);
        disp(['NumPerturbVisRep: ', num2str(NumPerturbVisRep)]);        
        disp(['Perturb Duration: NumPerturbVisRep * PerturbBasePatternDuration: ', num2str(NumPerturbVisRep * PerturbBasePatternDuration)]);
        %mod(PrePerturbDuration, PerturbBasePatternDuration)
    end

    % generate gray 'filler' frames using the number of frames required
    % after the last grating pattern to the end of vis stim (trained)
    %FullVidDuration = 
    % GrayFillerDuration = PrePerturbDuration - (PerturbBasePatternDuration * NumPerturbVisRep);
    % GrayFillerFrames = convergent(FramesPerSecond * GrayFillerDuration); % rounds ties to the nearest even integer
    % if (mod(GrayFillerFrames, 2) ~= 0)
    %     GrayFillerFrames = GrayFillerFrames + 1; % round up to nearest even integer
    % end
    % VideoGrayFiller = [repmat(GrayPattern, 1, GrayFillerFrames/2)]; % if no filler frames needed, then this will be empty

    %FullVideo = [];
    %FullVideo = VideoData;

    % NumPerturbVisRep = 20;
    %NumPerturbVisRep = S.GUI.NumPerturbVisRep;    
    %NumPerturbVisRep = S.GUI.NumISIOrigRep;
    %NumExtraPerturbVisRep = 50; % extra perturb patterns so video can continue till ITI for naive level
    
    % if debug
    %     NumExtraPerturbVisRep = 15;
    % end

    % use extra perturbation video for naive to extend vis stim to ITI
    switch S.GUI.TrainingLevel
        case 1 % Naive       
            %disp('Extra Vis Stim Repititions for Training Stage - Naive');
            NumExtraPerturbVisRep = 90;        
        case 2 % Mid Trained 1
            NumExtraPerturbVisRep = 90;
        case 3 % Mid Trained 2
            NumExtraPerturbVisRep = 0;
        case 4 % Trained
            %disp('No Extra Vis Stim Repititions for Training Stage - Trained');
            NumExtraPerturbVisRep = 0;
    end

    NumInitialBaseFrames = GratingFrames + GrayFixedFrames;
    NumPerturbBaseFrames = GrayPerturbFrames + GratingFrames;

    NumPerturbReps = NumPerturbVisRep + NumExtraPerturbVisRep;

    % TotalFramesNeeded = 2*(S.GUI.NumISIOrigRep * NumInitialBaseFrames) + ...
    %     GratingFrames; % number of frames needed is 
    TotalFramesNeeded = (S.GUI.NumISIOrigRep * NumInitialBaseFrames) + ...
        (S.GUI.NumISIOrigRep * NumInitialBaseFrames) * S.GUI.PostPerturbDurMultiplier + ...
        GratingFrames; % number of frames needed is 
    GrayFillerFramesNeeded = TotalFramesNeeded - (length(VideoData) + (length(VideoPerturbBasePattern)*NumPerturbReps));
    
    GrayFillerFramesNeeded = round(GrayFillerFramesNeeded); % get integer number of frames
    if (mod(GrayFillerFramesNeeded, 2) ~= 0)
        GrayFillerFramesNeeded = GrayFillerFramesNeeded + 1; % round up to nearest even integer
    end

    VideoGrayFiller = [repmat(GrayPattern, 1, GrayFillerFramesNeeded/2)];

    if ShowDebugOutput
        disp(['S.GUI.PostPerturbDurMultiplier: ', num2str(S.GUI.PostPerturbDurMultiplier)]);
        disp(['NumPerturbReps: ', num2str(NumPerturbReps)]);
        disp(['TotalFramesNeeded: ', num2str(TotalFramesNeeded)]);        
        disp(['GrayFillerFramesNeeded: ', num2str(GrayFillerFramesNeeded)]);
        disp(['length(VideoData) + (length(VideoPerturbBasePattern)*NumPerturbReps): ', num2str(length(VideoData) + (length(VideoPerturbBasePattern)*NumPerturbReps))]);
    end
    
    %NumPerturbReps = NumPerturbVisRep + 50;
    FullVideo = [VideoData repmat(VideoPerturbBasePattern, 1, NumPerturbReps) VideoGrayFiller GrayBlank]; % construct full video from initial and perturbation segments
                                                                                    %  and add final frame of grayscreen at the end of video
    % for j = 1:(NumPerturbVisRep + NumExtraPerturbVisRep)
    %     FullVideo = [FullVideo GrayPerturbVideo GratingVideo];
    % end    
    % FullVideo = [FullVideo GrayBlank]; % add final frame of grayscreen at the end of video   

    
    
    %FullVideo = [VideoData GrayPerturbVideo GratingVideo GrayPerturbVideo GratingVideo GrayPerturbVideo GratingVideo GrayPerturbVideo GratingVideo GrayPerturbVideo GratingVideo GrayBlank];
    %FullVideoFrames = S.GUI.NumISIOrigRep * (GratingFrames + GrayFixedFrames) + 3 * GratingFrames + 2 * GrayPerturbFrames + 1 + 3 * (GratingFrames + GrayPerturbFrames); % + 1 for final frame
    % FullVideoFrames = S.GUI.NumISIOrigRep * (GratingFrames + GrayFixedFrames) + GratingFrames + NumPerturbVisRep * (GrayPerturbFrames + GratingFrames) + 1; % + GratingFrame for the grating between base and variable segments of video and + 1 for final frame
    
    %FullVideoFrames = S.GUI.NumISIOrigRep * (GratingFrames + GrayFixedFrames) + GratingFrames + NumPerturbVisRep * (GrayPerturbFrames + GratingFrames) + NumExtraPerturbVisRep * (GrayPerturbFrames + GratingFrames) + 1; % + GratingFrame for the grating between base and variable segments of video and + 1 for final frame
    
    if GrayFillerFramesNeeded < 0
        GrayFillerFramesNeeded = 0;
    end

    % FullVideoFrames = S.GUI.NumISIOrigRep * NumInitialBaseFrames + ...
    %     GratingFrames + ...
    %     NumPerturbVisRep * NumPerturbBaseFrames + ...
    %     NumExtraPerturbVisRep * NumPerturbBaseFrames + ...
    %     1; % + GratingFrame for the grating between base and variable segments of video and + 1 for final frame

    FullVideoFrames = S.GUI.NumISIOrigRep * NumInitialBaseFrames + ...
        GratingFrames + ...
        NumPerturbVisRep * NumPerturbBaseFrames + ...
        NumExtraPerturbVisRep * NumPerturbBaseFrames + ...
        GrayFillerFramesNeeded + ...
        1; % + GratingFrame for the grating between base and variable segments of video and + 1 for final frame

    if ShowDebugOutput
        disp(['FullVideoFrames: (calculated): ', num2str(FullVideoFrames)]);
    end

    FullVideoFrames = length(FullVideo);
    
    if ShowDebugOutput
        disp(['FullVideoFrames: length(FullVideo): ', num2str(length(FullVideo))]);
    end

    %VideoStartToGoCueFrames = S.GUI.NumISIOrigRep * (GratingFrames + GrayFixedFrames) + NumPerturbVisRep * (GrayPerturbFrames + GratingFrames) + GratingFrames + 1; % 2*NumISIOrigRep for the repeated grating-gray pattern until go cue, + GratingFrame for the grating between base and variable segments of video and + 1 for final frame
    
    % FullVideoFrames = S.GUI.NumISIOrigRep * NumInitialBaseFrames + ...
    %     GratingFrames + ...
    %     NumPerturbVisRep * NumPerturbBaseFrames + ...
    %     1;
    
    % switch S.GUI.TrainingLevel
    %     case 1 % Naive
    %         VideoStartToGoCueFrames = S.GUI.NumISIOrigRep * NumInitialBaseFrames + ...
    %             GratingFrames + ...    
    %             NumPerturbVisRep * NumPerturbBaseFrames + ...                
    %             1; % 2*NumISIOrigRep for the repeated grating-gray pattern until go cue, + GratingFrame for the grating between base and variable segments of video and + 1 for final frame            
    %     case 2 % Trained                                                                    
    %         VideoStartToGoCueFrames = 2*(S.GUI.NumISIOrigRep * NumInitialBaseFrames) + ...
    %             GratingFrames + ...
    %             1; % 2*NumISIOrigRep for the repeated grating-gray pattern until go cue, + GratingFrame for the grating between base and variable segments of video and + 1 for final frame
    % end

    % VideoStartToGoCueFrames = 2*(S.GUI.NumISIOrigRep * NumInitialBaseFrames) + ...
    %     GratingFrames + ...
    %     1; % 2*NumISIOrigRep for the repeated grating-gray pattern until go cue, + GratingFrame for the grating between base and variable segments of video and + 1 for final frame
   
    VideoStartToGoCueFrames = S.GUI.NumISIOrigRep * NumInitialBaseFrames + ...
        NumPerturbVisRep * NumPerturbBaseFrames + ...
        GratingFrames + ...
        GrayFillerFramesNeeded + ...
        1; % 2*NumISIOrigRep for the repeated grating-gray pattern until go cue, + GratingFrame for the grating between base and variable segments of video and + 1 for final frame    

    % VideoStartToGoCueFrames = 2*(S.GUI.NumISIOrigRep * NumInitialBaseFrames) + ...
    % GratingFrames + ...
    % GrayFillerFrames + ...
    % 1;

    if ShowDebugOutput
        disp(['VideoStartToGoCueFrames: ', num2str(VideoStartToGoCueFrames)]);
        disp(['S.GUI.NumISIOrigRep: ', num2str(S.GUI.NumISIOrigRep)]);
        disp(['NumInitialBaseFrames: ', num2str(NumInitialBaseFrames)]);
        disp(['S.GUI.NumISIOrigRep * NumInitialBaseFrames: ', num2str(S.GUI.NumISIOrigRep * NumInitialBaseFrames)]);
        disp(['NumPerturbVisRep: ', num2str(NumPerturbVisRep)]);
        disp(['NumPerturbBaseFrames: ', num2str(NumPerturbBaseFrames)]);
        disp(['NumPerturbVisRep * NumPerturbBaseFrames: ', num2str(NumPerturbVisRep * NumPerturbBaseFrames)]);
        disp(['GratingFrames: ', num2str(GratingFrames)]);
        disp(['GrayFillerFramesNeeded: ', num2str(GrayFillerFramesNeeded)]);
    end

    VisStimDuration = VideoStartToGoCueFrames / FramesPerSecond; % calculate duration of vis stim based on when go cue state should begin after
                                                                    % start of vis stim state

    if ShowDebugOutput
        disp(['VisStimDuration (to go cue): ', num2str(VisStimDuration)]);
        disp(['VisStimDuration (valid for Naive) (FullVideoFrames/FramesPerSecond): ', num2str(FullVideoFrames/FramesPerSecond)]);
        disp(['FullVideoFrames: ', num2str(FullVideoFrames)]);
        disp(['length(FullVideo): ', num2str(length(FullVideo))]);
    end

    % load constructed video into the video object
    BpodSystem.PluginObjects.V.Videos{5} = struct;
    BpodSystem.PluginObjects.V.Videos{5}.nFrames = FullVideoFrames; 
    BpodSystem.PluginObjects.V.Videos{5}.Data = FullVideo;
    
    %% Generate audio stim based on vis stim for this trial
    %AudioStimSound = GenerateSineWave(SF, S.GUI.AudioStimFreq_Hz, S.GUI.GratingDur_s); % Sampling freq (hz), Sine frequency (hz), duration (s)
    
    %H.load(4, AudioStimSound);
    
    %VideoPrePerturbPattern = [GratingVideo GrayVideo]; % base video pattern for initial segment repetitions of grating->gray
    %VideoData = [repmat(VideoPrePerturbPattern, 1, S.GUI.NumISIOrigRep) GratingVideo]; % construct initial video segment, add one more grating video after initial repetitions
        
    % build audio stim to match grating/gray pattern
    PrePerturbNoSoundOffset = 1102;
    PrePerturbNoSoundOffset = 0;

    PostPerturbNoSoundOffset = 2205;
    PostPerturbNoSoundOffset = 0;

    %GratingDur = length(GratingVideo) * (1/FramesPerSecond); % get duration of grating in number of audio samples for period of grating 
    GratingNumSamples = GratingDur * SF;     
    %GrayDur = length(GrayVideo) * (1/FramesPerSecond);
    GrayNumSamples = GrayDur * SF;  % get duration of gray in number of audio samples for period between audio stim

    NoSoundPrePerturb = zeros(1, GrayNumSamples+PrePerturbNoSoundOffset);

    %VideoData = [repmat(VideoPrePerturbPattern, 1, S.GUI.NumISIOrigRep) GratingVideo]; % construct initial video segment, add one more grating video after initial repetitions
    
    AudioPrePerturbPattern = [AudioStimSound NoSoundPrePerturb];
    PrePerturbAudioData = [repmat(AudioPrePerturbPattern, 1, S.GUI.NumISIOrigRep) AudioStimSound]; % construct preperturb audio
    
    GrayPerturbDur = length(GrayPerturbVideo) * (1/FramesPerSecond);  % get duration of gray perturb
    GrayPerturbNumSamples = GrayPerturbDur * SF;    % get duration of gray perturb in number of audio samples for period of grating
    NoSoundPerturb = zeros(1, GrayPerturbNumSamples+PostPerturbNoSoundOffset);

    AudioPerturbBasePattern = [NoSoundPerturb AudioStimSound];
    % GrayFillerDur = length(GrayFiller) * (1/FramesPerSecond);  % get duration of gray filler
    % GrayFillerNumSamples = GrayFillerDur * SF;  % get duration of gray filler in number of audio samples
    FullAudioStimData = [PrePerturbAudioData repmat(AudioPerturbBasePattern, 1, NumPerturbReps)];

    switch S.GUI.TrainingLevel
        case 1
            VideoStartToGoCueDur_s = VideoStartToGoCueFrames * (1/FramesPerSecond);
            VideoStartToGoCueDur_NumAudioSamples = SF * VideoStartToGoCueDur_s;
            ShiftedGoCue = [zeros(1, VideoStartToGoCueDur_NumAudioSamples) GoCueSound zeros(1, length(FullAudioStimData) - VideoStartToGoCueDur_NumAudioSamples - length(GoCueSound))];
            FullAudioStimData = FullAudioStimData + ShiftedGoCue; % update with mixed signal
            %FullAudioStimData = [PrePerturbAudioData repmat(AudioPerturbBasePattern, 1, NumPerturbVisRep)];
        case 2
            VideoStartToGoCueDur_s = VideoStartToGoCueFrames * (1/FramesPerSecond);
            VideoStartToGoCueDur_NumAudioSamples = SF * VideoStartToGoCueDur_s;
            ShiftedGoCue = [zeros(1, VideoStartToGoCueDur_NumAudioSamples) GoCueSound zeros(1, length(FullAudioStimData) - VideoStartToGoCueDur_NumAudioSamples - length(GoCueSound))];
            FullAudioStimData = FullAudioStimData + ShiftedGoCue; % update with mixed signal
            %FullAudioStimData = [PrePerturbAudioData repmat(AudioPerturbBasePattern, 1, NumPerturbVisRep)];
        case 3
            % nothing 
            VideoStartToGoCueDur_s = VideoStartToGoCueFrames * (1/FramesPerSecond);
            % VideoStartToGoCueDur_NumAudioSamples = SF * VideoStartToGoCueDur_s;
            
            % ShiftedGoCue = [zeros(1, VideoStartToGoCueDur_NumAudioSamples) GoCueSound zeros(1, length(FullAudioStimData) - VideoStartToGoCueDur_NumAudioSamples - length(GoCueSound))];
            FullAudioStimData = [FullAudioStimData GoCueSound]; % concat go cue to end of audio stim for mid 2 training level
            % figure()
            % plot(FullAudioStimData)
        case 4
            % nothing 
    end

    % AudioStimSound_t = (1:length(AudioStimSound))/SF;
    % 
    % FullAudioStimData_t = (1:length(FullAudioStimData))/SF;
    % 
    % FullVideo_t = (1:length(FullVideo))/FramesPerSecond;
    % 
    % figure();
    % hold on; % one moment
    % plot(FullAudioStimData_t-(1/SF), FullAudioStimData);
    % plot(FullVideo_t-(1/60), FullVideo-14);
    % 
    % legend('Audio Signal','Video Data (Textures)');

    H.load(5, FullAudioStimData);
    %H.load(4, FullAudioStimData);

    % Query duration of one monitor refresh interval:
    ifi=Screen('GetFlipInterval', BpodSystem.PluginObjects.V.Window);
    
    % VideoPerturbBasePattern = [GrayPerturbVideo GratingVideo]; % perturbation video pattern for second segment repetitions of random ISI gray->grating
    % 
    % NumInitialBaseFrames = GratingFrames + GrayFixedFrames;
    % NumPerturbBaseFrames = GrayPerturbFrames + GratingFrames;
    % 
    % NumPerturbReps = NumPerturbVisRep + NumExtraPerturbVisRep;
    % TotalFramesNeeded = (S.GUI.NumISIOrigRep * NumInitialBaseFrames) + ...
    %     (S.GUI.NumISIOrigRep * NumInitialBaseFrames) * S.GUI.PostPerturbDurMultiplier + ...
    %     GratingFrames; % number of frames needed is 
    % GrayFillerFramesNeeded = TotalFramesNeeded - (length(VideoData) + (length(VideoPerturbBasePattern)*NumPerturbReps));
    % VideoGrayFiller = [repmat(GrayPattern, 1, GrayFillerFramesNeeded/2)];
    % FullVideo = [VideoData repmat(VideoPerturbBasePattern, 1, NumPerturbReps) VideoGrayFiller GrayBlank]; % construct full video from initial and perturbation segments


    % % debug 
    % pause(1);
    % 
    % global NumPosMissed;
    % NumPosMissed = 0;
    % BpodSystem.PluginObjects.V.play(5);
    % BpodSystem.PluginObjects.V.stop;
    % disp('NumPosMissed:');
    
    % disp(NumPosMissed);
    % PerceptualVBLSyncTest([screen=1][, stereomode=0][, fullscreen=1][, doublebuffer=1][, maxduration=10][, vblSync=1][, testdualheadsync=0][, useVulkan=0])



    %% update trial-specific Audio
    %H.DigitalAttenuation_dB = S.GUI.DigitalAttenuation_dB; % update sound level to param GUI
    if S.GUI.IncorrectSound
        OutputActionArgIncorrect = {'HiFi1', ['P' 2]};
    else
        OutputActionArgIncorrect = {};
    end

    %LastInitCueVolume = S.GUI.InitCueVolume_percent;
    % LastGoCueVolume = S.GUI.GoCueVolume_percent;
    % LastAudioStimVolume = S.GUI.AudioStimVolume_percent;
    % LastIncorrectSoundVolume = S.GUI.IncorrectSoundVolume_percent;

    if (S.GUI.InitCueFreq_Hz ~= LastInitCueFrequency) || ...
        (S.GUI.InitCueDuration_s ~= LastInitCueDuration) || ...
        (S.GUI.InitCueVolume_percent ~= LastInitCueVolume)
        InitCueSound = GenerateSineWave(SF, S.GUI.InitCueFreq_Hz, S.GUI.InitCueDuration_s)*S.GUI.InitCueVolume_percent; % Sampling freq (hz), Sine frequency (hz), duration (s)
        InitCueSound = ApplySoundEnvelope(InitCueSound, Envelope);
        H.load(1, InitCueSound);
        LastInitCueFrequency = S.GUI.InitCueFreq_Hz;
        LastInitCueDuration = S.GUI.InitCueDuration_s;
        LastInitCueVolume = S.GUI.InitCueVolume_percent;
    end
    if (S.GUI.GoCueFreq_Hz ~= LastGoCueFrequency) || ...
        (S.GUI.GoCueDuration_s ~= LastGoCueDuration) || ...
        (S.GUI.GoCueVolume_percent ~= LastGoCueVolume)
        GoCueSound = GenerateSineWave(SF, S.GUI.GoCueFreq_Hz, S.GUI.GoCueDuration_s)*S.GUI.GoCueVolume_percent; % Sampling freq (hz), Sine frequency (hz), duration (s)
        GoCueSound = ApplySoundEnvelope(GoCueSound, Envelope);
        H.load(2, GoCueSound);
        LastGoCueFrequency = S.GUI.GoCueFreq_Hz;
        LastGoCueDuration = S.GUI.GoCueDuration_s;
        LastGoCueVolume = S.GUI.GoCueVolume_percent;
    end
    % if (S.GUI.InitCueDuration_s ~= LastInitCueDuration)
    %     InitCueSound = GenerateSineWave(SF, S.GUI.InitCueFreq_Hz, S.GUI.InitCueDuration_s); % Sampling freq (hz), Sine frequency (hz), duration (s)        
    %     H.load(1, InitCueSound);
    %     LastInitCueDuration = S.GUI.InitCueDuration_s;
    % end
    % if S.GUI.GoCueDuration_s ~= LastGoCueDuration        
    %     GoCueSound = GenerateSineWave(SF, S.GUI.GoCueFreq_Hz, S.GUI.GoCueDuration_s); % Sampling freq (hz), Sine frequency (hz), duration (s)
    %     H.load(2, GoCueSound);
    %     LastGoCueDuration = S.GUI.GoCueDuration_s;
    % end
    if (S.GUI.PunishSoundDuration_s ~= LastPunishSoundDuration) || ...
        (S.GUI.IncorrectSoundVolume_percent ~= LastIncorrectSoundVolume)
        IncorrectSound = GenerateWhiteNoise(SF, S.GUI.PunishSoundDuration_s, 1, 2)*S.GUI.IncorrectSoundVolume_percent; % white noise punish sound
        IncorrectSound = ApplySoundEnvelope(IncorrectSound, Envelope);
        H.load(3, IncorrectSound);
        LastPunishSoundDuration = S.GUI.PunishSoundDuration_s;
        LastIncorrectSoundVolume = S.GUI.IncorrectSoundVolume_percent;
    end
     
    
    
    %% update trial-specific valve times according to set reward amount
    %R = GetValveTimes(S.GUI.RewardAmount, [1 3]); LeftValveTime = R(1); RightValveTime = R(2); % Update reward amounts
    
    % LeftValveTime = 0.02; CenterValveTime = 0.02; RightValveTime = 0.02;
    % LeftValveTime = 1; CenterValveTime = 0.5; RightValveTime = 1;
    % LeftValveTime = 0.25; CenterValveTime = 0.05; RightValveTime = 0.25;
    
    % LeftValveTime = S.GUI.LeftValveTime_s;
    % RightValveTime = S.GUI.RightValveTime_s;
    % CenterValveTime = S.GUI.CenterValveTime_s;

    %% update trial-specific state matrix fields

    % port 1: left (check on bPod console)
    % port 2: middle
    % port 3: right

    switch TrialTypes(currentTrial) % Determine trial-specific state matrix fields
        case 1 % short ISI (defined on line 730) % left side is rewarded
            CorrectLick = 'Port1In';
            CorrectPort = 'Port1';
            IncorrectLick = 'Port3In';
            IncorrectPort = 'Port3';
            StimulusOutput = {'SoftCode', 3};
            RewardPortOut = 'Port1Out'; ValveTime = LeftValveTime; Valve = 'Valve1';
            CorrectWithdrawalEvent = 'Port1Out';
            %VisualGratingDuration = S.GUI.LDuration; % length of time to play visual stimulus when Left TrialType

            ExperimenterTrialInfo.CorrectChoice = 'Left';   % capture variable states as field/value struct for experimenter info
        case 2  % long ISI % right side is rewarded
            CorrectLick = 'Port3In';
            CorrectPort = 'Port3';
            IncorrectLick = 'Port1In';
            IncorrectPort = 'Port1';
            StimulusOutput = {'SoftCode', 4};
            RewardPortOut = 'Port3Out'; ValveTime = RightValveTime; Valve = 'Valve3';
            CorrectWithdrawalEvent = 'Port3Out';
            %VisualGratingDuration = S.GUI.RDuration; % length of time to play visual stimulus when Right TrialType

            ExperimenterTrialInfo.CorrectChoice = 'Right';   % capture variable states as field/value struct for experimenter info
    end

    %% set variables for trial-specific training level   
    % if S.GUI.TrainingLevel == 1 % Reward both sides (overriding switch/case above)
    %     RightActionState = 'Reward'; LeftActionState = 'Reward';
    % elseif S.GUI.TrainingLevel == 2
    %     disp('Medium');
    % elseif S.GUI.TrainingLevel == 3
    %     disp('Hard');
    % end
    
    %% trial-specific output actions
    % maybe move this to state-matrix section for consistency
    OutputActionArgInitCue = {'HiFi1', ['P' 0], 'BNCState', 1};
    if S.GUI.VisStimEnable && S.GUI.AudioStimEnable % both vis and audio stim enabled
        OutputActionArgGoCue = {'BNCState', 1};
        OutputActionAudioVisStim = {'SoftCode', 5, 'HiFi1', ['P' 4], 'BNCState', 1};
    elseif S.GUI.VisStimEnable % only vis stim enabled, need regular gocue at state
        OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};
        OutputActionAudioVisStim = {'SoftCode', 5};
    elseif S.GUI.AudioStimEnable
        OutputActionArgGoCue = {'BNCState', 1};
        OutputActionAudioVisStim = {'HiFi1', ['P' 4], 'BNCState', 1};
    else
        OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};
        OutputActionAudioVisStim = {};
    end
    %OutputActionsCenterLick = {'HiFi1', 'X'}; % stop audio stim
    %OutputActionsPreGoCueDelay = {'HiFi1', 'X'}; % stop audio stim
    OutputActionsPreGoCueDelay = {}; % 
    OutputActionsEarlyChoice = {'SoftCode', 255, 'HiFi1', 'X'}; % stop audio stim, stop vis stim
    OutputActionsPunishSetup = {'SoftCode', 255, 'HiFi1', 'X'};

    %% training level specific state matrix values

    switch S.GUI.TrainingLevel
        case 1 % naive        
            if ShowDebugOutput
                disp('Training Stage - Naive');
            end
            InitCue_Tup_NextState = 'InitWindow'; % change to fn_lic_poiss_3              
            %DidNotChoose_Tup_NextState = 'Reward';
            DidNotChoose_Tup_NextState = 'ITI';
        
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterLick'};
            VisualStimulusStateChangeConditions = {'Tup', 'CenterReward'};
    
            %PreGoCueDelay_OutputActions = {}; % no output action in PreGoCueDelay for naive
            CenterReward_OutputActions = {'Valve2', 1};

            GoCue_Tup_NextState = 'RewardNaive';  % naive
            %OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};
            OutputActionAudioVisStim = {'SoftCode', 5, 'HiFi1', ['P' 4], 'BNCState', 1};
            WindowChoice_StateChangeConditions = {};
            OutputActionsWindowChoice = {};
            Reward_Tup_NextState = 'ITI';
            PunishSetup_Tup_NextState = 'PunishNaive'; % Naive
            WindCenterEvents = {'Tup', 'DidNotLickCenter', 'Port2In', 'CenterReward', 'Condition5', 'CenterReward'};


            ExperimenterTrialInfo.TrainingLevel = 'Naive';
        case 2 % Mid 1 Trained
            if ShowDebugOutput
                disp('Training Stage - Mid Trained 1');
            end
            InitCue_Tup_NextState = 'InitWindow';
            DidNotChoose_Tup_NextState = 'ITI';
    
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterReward', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterReward', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterLick'};
            VisualStimulusStateChangeConditions = {'Tup', 'CenterReward'};
                
            % PreGoCueDelay_OutputActions = {'SoftCode', 255}; % stop vis stim in PreGoCueDelay so its init and perturb segment durations are equal for well trained
            %CenterReward_OutputActions = {'Valve2', 1, 'SoftCode', 255}; % moved video stop code earlier to center reward
            CenterReward_OutputActions = {'Valve2', 1}; % moved video stop code earlier to center reward

            GoCue_Tup_NextState = 'WindowChoice';  % trained
            %OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};
            OutputActionAudioVisStim = {'SoftCode', 5, 'HiFi1', ['P' 4], 'BNCState', 1};
            WindowChoice_StateChangeConditions = {CorrectLick, 'RewardDelay', 'Condition1', 'RewardDelay', IncorrectLick, 'PunishSetup', 'Condition2', 'PunishSetup', 'Tup', 'DidNotChoose'};
            OutputActionsWindowChoice = {};
            %OutputActionsWindowChoice = {'HiFi1', 'X'};
            Reward_Tup_NextState = 'ExtraStimDurPostRew_Naive';
            PunishSetup_Tup_NextState = 'Punish'; % trained
            WindCenterEvents = {'Tup', 'DidNotLickCenter', 'Port2In', 'CenterReward', 'Condition5', 'CenterReward'};


            ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 1';
        case 3 % Mid 2 Trained
            if ShowDebugOutput
                disp('Training Stage - Mid Trained 2');
            end
            InitCue_Tup_NextState = 'InitWindow';
            DidNotChoose_Tup_NextState = 'ITI';
    
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterReward', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterReward', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterLick'};
            VisualStimulusStateChangeConditions = {'Tup', 'CenterReward'};                

            % PreGoCueDelay_OutputActions = {'SoftCode', 255}; % stop vis stim in PreGoCueDelay so its init and perturb segment durations are equal for well trained
            CenterReward_OutputActions = {'Valve2', 1, 'SoftCode', 255}; % moved video stop code earlier to center reward

            GoCue_Tup_NextState = 'WindowChoice';  % trained
            %OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};
            OutputActionAudioVisStim = {'SoftCode', 5, 'HiFi1', ['P' 4], 'BNCState', 1};
            WindowChoice_StateChangeConditions = {CorrectLick, 'RewardDelay', 'Condition1', 'RewardDelay', IncorrectLick, 'PunishSetup', 'Condition2', 'PunishSetup', 'Tup', 'DidNotChoose'};
            OutputActionsWindowChoice = {'HiFi1', 'X'};
            Reward_Tup_NextState = 'ITI';
            PunishSetup_Tup_NextState = 'Punish'; % trained
            WindCenterEvents = {'Tup', 'DidNotLickCenter', 'Port2In', 'CenterReward', 'Condition5', 'CenterReward'};


            ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 2';
        case 4 % well trained
            if ShowDebugOutput
                disp('Training Stage - Trained');
            end
            InitCue_Tup_NextState = 'InitWindow';
            DidNotChoose_Tup_NextState = 'ITI';
    
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterReward', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterReward', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterLick'};
            VisualStimulusStateChangeConditions = {'Tup', 'CenterReward'};
                
            % PreGoCueDelay_OutputActions = {'SoftCode', 255}; % stop vis stim in PreGoCueDelay so its init and perturb segment durations are equal for well trained
            CenterReward_OutputActions = {'Valve2', 1, 'SoftCode', 255}; % moved video stop code earlier to center reward

            GoCue_Tup_NextState = 'WindowChoice';  % trained
            OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};
            OutputActionAudioVisStim = {'SoftCode', 5, 'HiFi1', ['P' 4], 'BNCState', 1};
            WindowChoice_StateChangeConditions = {CorrectLick, 'CorrectLickInterval', IncorrectLick, 'IncorrectLickInterval', 'Tup', 'DidNotChoose'};
            OutputActionsWindowChoice = {'HiFi1', 'X'};
            Reward_Tup_NextState = 'ITI';
            PunishSetup_Tup_NextState = 'Punish'; % trained
            ExperimenterTrialInfo.TrainingLevel = 'Well Trained';
            WindCenterEvents = {'Tup', 'DidNotLickCenter', 'Port2In', 'CenterReward', 'Condition5', 'CenterReward', 'Port1In', 'EarlyChoiceDurCenterLick', 'Port3In', 'EarlyChoiceDurCenterLick'};
    end

    %% add console print for experimenter trial information
    %disp(['currentTrial: ', num2str(currentTrial)]);

    %ExperimenterTrialInfo = struct()
    strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
    disp(strExperimenterTrialInfo);



    %% set variables for trial-specific difficulty level       
    % modify response time allowance according to set difficulty level
    % EasyPercentage = 1.00;
    % MediumPercentage = 0.75;
    % HardPercentage = 0.50;
    % if TrialDifficulty == 2
    %     disp('Difficulty - Medium');
    %     InitWindowTimeout = S.GUI.InitWindowTimeout_s * MediumPercentage;
    %     ChoiceWindow = S.GUI.ChoiceWindow_s * MediumPercentage;    
    % 
    %     ConfirmLickInterval = S.GUI.ConfirmLickInterval_s;  % need to increase this var to increase difficulty
    %     PunishSoundDuration = S.GUI.PunishSoundDuration_s;  % need to increase this var to increase difficulty
    % 
    %     ChoiceConfirmWindow = S.GUI.ChoiceConfirmWindow_s  * MediumPercentage;
    %     ChoiceReattemptWindow = S.GUI.ChoiceReattemptWindow  * MediumPercentage;
    % elseif TrialDifficulty == 3
    %     disp('Difficulty - Hard');
    %     InitWindowTimeout = S.GUI.InitWindowTimeout_s * HardPercentage;
    %     ChoiceWindow = S.GUI.ChoiceWindow_s * HardPercentage;
    % 
    %     ChoiceConfirmWindow = S.GUI.ChoiceConfirmWindow_s  * HardPercentage;
    %     ChoiceReattemptWindow = S.GUI.ChoiceReattemptWindow  * HardPercentage;        
    % else
    %     disp('Difficulty - Easy');
    %     InitWindowTimeout = S.GUI.InitWindowTimeout_s * EasyPercentage;
    %     ChoiceWindow = S.GUI.ChoiceWindow_s * EasyPercentage;
    % 
    %     ChoiceConfirmWindow = S.GUI.ChoiceConfirmWindow_s  * EasyPercentage;
    %     ChoiceReattemptWindow = S.GUI.ChoiceReattemptWindow  * EasyPercentage;        
    % end

    % S.GUI.InitWindowTimeout_s = S.GUI.InitWindowTimeout_s;
    % S.GUI.ChoiceWindow_s = S.GUI.ChoiceWindow_s;
    % S.GUI.ConfirmLickInterval_s = S.GUI.ConfirmLickInterval_s;
    % S.GUI.PunishSoundDuration_s = S.GUI.PunishSoundDuration_s;
    % S.GUI.ChoiceConfirmWindow_s = S.GUI.ChoiceConfirmWindow_s;
    %ChoiceReattemptWindow = S.GUI.ChoiceReattemptWindow;



    %% construct state matrix
    sma = NewStateMatrix(); % Assemble state matrix
    % needs more conditions here when adding difficulty levels
    sma = SetCondition(sma, 1, CorrectPort, 1); % Condition 1: Correct Port is high (licking)
    sma = SetCondition(sma, 2, IncorrectPort, 1); % Condition 2: Incorrect Port is high (licking)

    % testing behavior port4in port4out for use as sync signal from camera
    sma = SetCondition(sma, 3, 'Port4', 1); % Condition 3: Sync signal high
    sma = SetCondition(sma, 4, 'Port4', 0); % Condition 4: Sync signal low

    sma = SetCondition(sma, 5, 'Port2', 1); % Condition 5: center port is high (licking)

    %disp('-------------------------------------------------------'); % barrier for debug info

    % start state machine for this trial, push updated sounds (if any) to
    % HiFi
    sma = AddState(sma, 'Name', 'Start', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'InitCue'},...
        'OutputActions', {'HiFi1','*'}); % Code to push newly uploaded waves to front (playback) buffers
    % play init cue sound
    sma = AddState(sma, 'Name', 'InitCue', ...
        'Timer', S.GUI.InitCueDuration_s,...
        'StateChangeConditions', {'Tup', InitCue_Tup_NextState},...         
        'OutputActions', OutputActionArgInitCue);

    % naive - center spout reward to condition init licking
    % maybe give this it's own unique valve time?
    sma = AddState(sma, 'Name', 'InitReward', ...
        'Timer', CenterValveTime,...
        'StateChangeConditions', {'Tup', 'InitWindow'},...
        'OutputActions', {'Valve2', 1});

    % outdated comment, update
    % naive - auto go to vis stimulus after playing go cue sound (we still
    %         log DidNotInitiate, but go to vis stim instead of ITI)
    % trained - wait for lick after playing go cue sound
    sma = AddState(sma, 'Name', 'InitWindow', ...
        'Timer', S.GUI.InitWindowTimeout_s,...
        'StateChangeConditions', {'Tup', 'InitCueAgain', 'Port2In', 'PreVisStimDelay', 'Port1In', 'WrongInitiation', 'Port3In', 'WrongInitiation'},...         
        'OutputActions', {});    
        
    sma = AddState(sma, 'Name', 'InitCueAgain', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'InitCue'},...         
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'WrongInitiation', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...         
        'OutputActions', {});     

    % note: stay in trial until init
    % naive - log DidNotInitiate, and go to vis stim 
    % trained - log DidNotInitiate, go to ITI
    % sma = AddState(sma, 'Name', 'DidNotInitiate', ...
    %     'Timer', 0,...
    %     'StateChangeConditions', {'Tup', DidNotInitiate_Tup_NextState},...         
    %     'OutputActions', {});    
    
    % delay after init before visual stimulus
    % placeholder - not in  use
    sma = AddState(sma, 'Name', 'PreVisStimDelay', ...
        'Timer', S.GUI.PreVisStimDelay_s,...
        'StateChangeConditions', {'Tup', 'VisualStimulus'},...
        'OutputActions', {});    
    
    % VisualStimulus
    sma = AddState(sma, 'Name', 'VisualStimulus', ...
        'Timer', VisStimDuration,...
        'StateChangeConditions', VisualStimulusStateChangeConditions,...
        'OutputActions', OutputActionAudioVisStim);
      %'Timer', VisStimDuration,...

    % Well trained: after the stimulus ends, mouse center licks, then center reward happens.  
    % sma = AddState(sma, 'Name', 'CenterLick', ...
    %     'Timer', S.GUI.WindCenterLick_s,...
    %     'StateChangeConditions', WindCenterEvents,...
    %     'OutputActions', {}); 

    sma = AddState(sma, 'Name', 'CenterReward', ...
        'Timer', CenterValveTime,...
        'StateChangeConditions', {'Tup', 'PreGoCueDelay'},...
        'OutputActions', CenterReward_OutputActions);    
    %'OutputActions', {'Valve2', 1});
    %{'SoftCode', 255} % in case needed to stop video if this is used
    %outside of naive

    
    % placeholder - not in  use
    sma = AddState(sma, 'Name', 'PreGoCueDelay', ...
        'Timer', S.GUI.PreGoCueDelay_s,...
        'StateChangeConditions', {'Tup', 'GoCue', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'},...         
        'OutputActions', OutputActionsPreGoCueDelay);
   
    %'OutputActions', PreGoCueDelay_OutputActions);

    % also stop video here, there can be small delay even with zero timer
    % when waiting to ITI
    sma = AddState(sma, 'Name', 'EarlyChoice', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', OutputActionsEarlyChoice);

    % well trained - mouse didn't lick center after vis stim during S.GUI.CenterLickWindow_Trained_s
    sma = AddState(sma, 'Name', 'EarlyChoiceDurCenterLick', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});
    
    % play go cue sound
    sma = AddState(sma, 'Name', 'GoCue', ...
        'Timer', S.GUI.GoCueDuration_s,...
        'StateChangeConditions', {'Tup', GoCue_Tup_NextState},...         
        'OutputActions', OutputActionArgGoCue);      
    %'OutputActions', {'BNCState', 1});    % maybe, although timing could be off for sync sig
        
    %
    %'OutputActions', {'BNCState', 1});    % maybe, although timing could be off for sync sig

    sma = AddState(sma, 'Name', 'RewardNaive', ...
        'Timer', CenterValveTime,...
        'StateChangeConditions', {'Tup', 'WindowRewardGrab_Naive'},...
        'OutputActions', {Valve, 1});
%'Timer', ValveTime,...
    
    
    % keep valve open here? no
    sma = AddState(sma, 'Name', 'WindowRewardGrab_Naive', ...
        'Timer', S.GUI.WindowRewardGrabDuration_Naive_s,...
        'StateChangeConditions', {'Tup', 'DidNotChoose', CorrectLick, 'ExtraStimDurPostRew_Naive', IncorrectLick, 'PunishSetup'},...
        'OutputActions', {});
    
    % additional time to associate stimulus with reward
    sma = AddState(sma, 'Name', 'ExtraStimDurPostRew_Naive', ...
        'Timer', S.GUI.ExtraStimDurPostRew_Naive_s,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});

   %'Timer', S.GUI.WindowRewardGrabDuration_Naive_s,...
    
    % naive - don't require confirm/choose again loop, log DidNotChoose,
    %         reward from correct spout anyway
    % trained - 
    % well trained - ?
    sma = AddState(sma, 'Name', 'WindowChoice', ...
        'Timer', S.GUI.ChoiceWindow_s,...
        'StateChangeConditions', WindowChoice_StateChangeConditions,...
        'OutputActions', OutputActionsWindowChoice);        
        %'OutputActions', {}); 

        % 'StateChangeConditions', {'Port1In', LeftLickAction, 'Port3In', RightLickAction, 'Tup', 'ITI'},...
        % 'OutputActions', {'HiFi1', SoundOffBytes});
        %'OutputActions', {'SoftCode', 255, 'HiFi1', SoundOffBytes});
    
    % trained - did not lick center after vis stim for center reward
    sma = AddState(sma, 'Name', 'DidNotLickCenter', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});


    % naive - log DidNotChoose, skip confirm/choose again loop, go to
    %         reward correct spout
    % trained - log DidNotChoose, go to ITI
    % well trained - ?
    sma = AddState(sma, 'Name', 'DidNotChoose', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', DidNotChoose_Tup_NextState},...
        'OutputActions', {});   

    %RewardDelay = S.GUI.RewardDelay_s
    
    % RewardPortOut = 'Port1Out';
    % RewardPortOut = 'Port3Out';
    % ValveTime = LeftValveTime;
    % ValveTime = RightValveTime;
    % ValveState = 1;
    % ValveState = 4;

    sma = AddState(sma, 'Name', 'CorrectLickInterval', ...
        'Timer', S.GUI.ConfirmLickInterval_s,...
        'StateChangeConditions', {'Tup', 'WindowCorrectChoiceConfirm', IncorrectLick, 'IncorrectLickInterval'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'IncorrectLickInterval', ...
        'Timer', S.GUI.ConfirmLickInterval_s,...
        'StateChangeConditions', {'Tup', 'WindowIncorrectChoiceConfirm', CorrectLick, 'CorrectLickInterval'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'WindowCorrectChoiceConfirm', ...
        'Timer', S.GUI.ChoiceConfirmWindow_s,...
        'StateChangeConditions', {'Tup', 'DidNotConfirm', CorrectLick, 'RewardDelay', IncorrectLick, 'IncorrectLickInterval', 'Condition1', 'RewardDelay', 'Condition2', 'IncorrectLickInterval'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'WindowIncorrectChoiceConfirm', ...
        'Timer', S.GUI.ChoiceConfirmWindow_s,...
        'StateChangeConditions', {'Tup', 'DidNotConfirm', CorrectLick, 'CorrectLickInterval', IncorrectLick, 'PunishSetup', 'Condition1', 'CorrectLickInterval', 'Condition2', 'PunishSetup'},...
        'OutputActions', {});


    % sma = AddState(sma, 'Name', 'WindowCorrectChoiceConfirm', ...
    %     'Timer', ChoiceConfirmWindow,...
    %     'StateChangeConditions', {'Tup', 'ChoiceConfirmWindow', CorrectLick, 'RewardDelay', IncorrectLick, 'IncorrectLickInterval', 'Condition1', 'RewardDelay', 'Condition2', 'IncorrectLickInterval'},...
    %     'OutputActions', {});
    % 
    % sma = AddState(sma, 'Name', 'WindowIncorrectChoiceConfirm', ...
    %     'Timer', ChoiceConfirmWindow,...
    %     'StateChangeConditions', {'Tup', 'ChoiceConfirmWindow', CorrectLick, 'CorrectLickInterval', IncorrectLick, 'PunishSetup', 'Condition1', 'CorrectLickInterval', 'Condition2', 'PunishSetup'},...
    %     'OutputActions', {});

    % sma = AddState(sma, 'Name', 'ChoiceConfirmWindow', ...
    %     'Timer', ChoiceReattemptWindow,...
    %     'StateChangeConditions', {'Tup', 'DidNotConfirm', CorrectLick, 'CorrectLickInterval', IncorrectLick, 'IncorrectLickInterval'},...
    %     'OutputActions', {});
    % 
    sma = AddState(sma, 'Name', 'DidNotConfirm', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'RewardDelay', ...
        'Timer', S.GUI.RewardDelay_s,...
        'StateChangeConditions', {'Tup', 'Reward'},...
        'OutputActions', {});

    %'StateChangeConditions', {'Tup', 'Reward', RewardPortOut, 'CorrectEarlyWithdrawal'
    sma = AddState(sma, 'Name', 'Reward', ...
        'Timer', ValveTime,...
        'StateChangeConditions', {'Tup', Reward_Tup_NextState},...
        'OutputActions', {Valve, 1});

    sma = AddState(sma, 'Name', 'PunishSetup', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', PunishSetup_Tup_NextState},...
        'OutputActions', OutputActionsPunishSetup); % Set white noise waveform
        %'OutputActions', {'HiFi1',['W' 0]}); % Set white noise waveform

    sma = AddState(sma, 'Name', 'PunishNaive', ...
        'Timer', S.GUI.PunishSoundDuration_s,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', OutputActionArgIncorrect);
    %'Timer', S.GUI.WindowRewardGrabDuration_Naive_s,...

    sma = AddState(sma, 'Name', 'Punish', ...
        'Timer', S.GUI.PunishSoundDuration_s,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', OutputActionArgIncorrect);

    %'Timer', S.GUI.PunishDelay,... % call it punish or incorrect?

    % sma = AddState(sma, 'Name', 'CorrectEarlyWithdrawal', ...
    %     'Timer', 0,...
    %     'StateChangeConditions', {'Tup', 'ITI'},...
    %     'OutputActions', {});
    sma = AddState(sma, 'Name', 'ITI', ...
        'Timer', ITI,...
        'StateChangeConditions', {'Tup', '>exit'},...
        'OutputActions', {'SoftCode', 255, 'HiFi1', 'X'});   
   
    SendStateMachine(sma); % Send the state matrix to the Bpod device        
    RawEvents = RunStateMachine; % Run the trial and return events
   
    if ~isempty(fieldnames(RawEvents)) % If trial data was returned (i.e. if not final trial, interrupted by user)
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
        BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
        BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial); % Adds the trial type of the current trial to data
        UpdateSideOutcomePlot(TrialTypes, BpodSystem.Data, 1);
        %UpdateTotalRewardDisplay(S.GUI.RewardAmount, currentTrial);
        %UpdateTotalRewardDisplay(S.GUI.RewardAmount, currentTrial); % how to change this for the way we're doing it?
        if useStateTiming
            StateTiming();
        end
        SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file
    end
    HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
    if BpodSystem.Status.BeingUsed == 0 % If protocol was stopped, exit the loop
        BpodSystem.PluginObjects.V = [];
        BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session
        return
    end
end
BpodSystem.PluginObjects.V = [];
BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session


function UpdateSideOutcomePlot(TrialTypes, Data, isEndOfTrial)
% Determine outcomes from state data and score as the SideOutcomePlot plugin expects
global BpodSystem
% make sure all trial outcomes are included here for plot
if isfield(Data, 'nTrials')
    % if field nTrials does exist in Data struct, then we're at trial 2 or more 
    % we're either calling UpdateSideOutcomePlot during the trial or at the end of a trial 
    % if this is during a trial, the call is done to update the choice case
    % as these can change due to anti-bias intervention controls
    % in this case, Data.nTrials hasn't yet incremented so only the choice
    % case for the current trial will be seen to change
    % otherwise this is at the end of a trial, and will update the plot
    % with the trial outcome
    Outcomes = zeros(1,Data.nTrials);
    for x = 1:Data.nTrials
        if ~isnan(Data.RawEvents.Trial{x}.States.Reward(1))
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                disp(['Outcome: Reward']);
            end
            Outcomes(x) = 1;    % draws green circle on outcome plot
        elseif ~isnan(Data.RawEvents.Trial{x}.States.Punish(1))
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                disp(['Outcome: Punish']);
            end
            Outcomes(x) = 0;    % draws red circle on outcome plot
        elseif ~isnan(Data.RawEvents.Trial{x}.States.ExtraStimDurPostRew_Naive(1))
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                disp(['Outcome: RewardNaive']);
            end
            Outcomes(x) = 1;    % draws green circle on outcome plot
        elseif ~isnan(Data.RawEvents.Trial{x}.States.PunishNaive(1))
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                disp(['Outcome: PunishNaive']);
            end
            Outcomes(x) = 0;    % draws red circle on outcome plot
        elseif ~isnan(Data.RawEvents.Trial{x}.States.WrongInitiation(1))
            if (x == Data.nTrials && isEndOfTrial)
                disp(['Outcome: WrongInitiation']);
            end
            Outcomes(x) = 3;    % draws clear circle on outcome plot
        elseif ~isnan(Data.RawEvents.Trial{x}.States.EarlyChoice(1))
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                disp(['Outcome: EarlyChoice']);
            end
            Outcomes(x) = 3;    % draws clear circle on outcome plot
        elseif ~isnan(Data.RawEvents.Trial{x}.States.EarlyChoiceDurCenterLick(1))
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                disp(['Outcome: EarlyChoiceDurCenterLick']);
            end
            Outcomes(x) = 3;    % draws clear circle on outcome plot
        elseif ~isnan(Data.RawEvents.Trial{x}.States.DidNotChoose(1))
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                disp(['Outcome: DidNotChoose']);
            end
            Outcomes(x) = 3;    % draws clear circle on outcome plot
        elseif ~isnan(Data.RawEvents.Trial{x}.States.DidNotConfirm(1))
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                disp(['Outcome: DidNotConfirm']);
            end
            Outcomes(x) = 3;    % draws clear circle on outcome plot
        elseif ~isnan(Data.RawEvents.Trial{x}.States.DidNotLickCenter(1))
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                disp(['Outcome: DidNotLickCenter']);
            end
            Outcomes(x) = 3;    % draws clear circle on outcome plot
        else
            % this is 'catch-all' to indicate that none of the above outcomes
            % occured so we know that we need to find/add more outcomes to list
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured          
                disp(['Outcome: Other']);
            end
            Outcomes(x) = 3;    % draws clear circle on outcome plot
        end
        if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
            %disp(['Data.nTrials:', num2str(Data.nTrials)]);
            disp('-------------------------------------------------------'); % visual barrier for experimenter info
        end
    end
    SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'update',Data.nTrials+1,2-TrialTypes,Outcomes);
else
    % if field nTrials doesn't exist in Data struct, then this is the first
    % trial so we only want SideOutcomePlot to update the trial choice case 
    Outcomes = zeros(1, 1);
    SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'update', 1,2-TrialTypes, Outcomes);
end

function UpdateTotalRewardDisplay(RewardAmount, currentTrial)
% If rewarded based on the state data, update the TotalRewardDisplay
global BpodSystem
    if ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial}.States.Reward(1))
        TotalRewardDisplay('add', RewardAmount);
    end

function [TrialTypes] = AdjustMaxConsecutiveSameSideWarmupTrials(TrialTypes, NumEasyWarmupTrials)       
    % modify trial types so that there are no more than 3 consecutive same
    % types
    MaxSameConsecutiveTrials = 3;
    %NewTrialTypes = TrialTypes;
    for i = MaxSameConsecutiveTrials:NumEasyWarmupTrials 
        if (i > MaxSameConsecutiveTrials)
            PrevMaxTrials = TrialTypes(i-3:i-1);
            if (all(PrevMaxTrials == 1) || all(PrevMaxTrials == 2))
                NewSameAsPrevMax = true;
                while NewSameAsPrevMax
                    DrawTrialType = unidrnd(2,1,1);       
                    if ~all(PrevMaxTrials == DrawTrialType)
                        NewSameAsPrevMax = false;
                    end
                end
                TrialTypes(i) = DrawTrialType;
            end
        end   
    end

% generate full envelope for sound given the sound and front part of
% envelope, return enveloped sound
function [SoundWithEnvelope] = ApplySoundEnvelope(Sound, Envelope)
    BackOfTheEnvelope = fliplr(Envelope);   % flipe front envelope to get back envelope
    IdxsBetweenTheEnvelope = length(Sound) - 2 * length(Envelope); % indices between front and back of envelope
    FullEnvelope = [Envelope ones(1, IdxsBetweenTheEnvelope) BackOfTheEnvelope];  % full envelope
    SoundWithEnvelope = Sound .* FullEnvelope;    % apply envelope element-wise