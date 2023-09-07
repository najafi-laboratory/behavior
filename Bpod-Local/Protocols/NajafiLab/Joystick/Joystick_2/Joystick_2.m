function Joystick_2      
global BpodSystem

global ShowDebugOutput;
ShowDebugOutput = 1;

global ForceITIZero; % to speed up testing
ForceITIZero = 1;
%% Assert Rotary Encoder module is present + USB-paired (via USB button on console GUI)
BpodSystem.assertModule('RotaryEncoder', 1);
% Create an instance of the RotaryEncoder module
R = RotaryEncoderModule(BpodSystem.ModuleUSB.RotaryEncoder1); 
% Ensure Rotary Encoder module is version 2
if BpodSystem.Modules.HWVersion_Major(strcmp(BpodSystem.Modules.Name, 'RotaryEncoder1')) < 2
    error('Error: This protocol requires rotary encoder module v2 or newer');
end

%% Assert HiFi module is present + USB-paired (via USB button on console GUI)
BpodSystem.assertModule('HiFi', 1); % The second argument (1) indicates that the HiFi module must be paired with its USB serial port
% Create an instance of the HiFi module
H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1); % The argument is the name of the HiFi module's USB serial port (e.g. COM3)

%% Setup (runs once before the first trial)
% MaxTrials = 5000; % Set to some sane value, for preallocation

%--- Define parameters and trial structure
S = BpodSystem.ProtocolSettings; % Loads settings file chosen in launch manager into current workspace as a struct called 'S'
if isempty(fieldnames(S))  % If chosen settings file was an empty struct, populate struct with default settings
    % Define default settings here as fields of S (i.e S.InitialDelay = 3.2)
    % Note: Any parameters in S.GUI will be shown in UI edit boxes. 
    % See ParameterGUI plugin documentation to show parameters as other UI types (listboxes, checkboxes, buttons, text)
    
    %% ITI params
    S.GUI.ITImin_s = 1;    % Minimum ITI (in seconds)
    S.GUI.ITImax_s = 5;    % Maximum ITI (in seconds)
    S.GUI.ITIlambda = 0.3;  % ITIlambda parameter of the exponential distribution
    S.GUIPanels.ITI_Dist = {'ITImin_s', 'ITImax_s', 'ITIlambda'};

    %% sound params
    S.GUI.DigitalAttenuation_dB = -40; % volume control: H.DigitalAttenuation_dB = -40;
    %S.GUI.AmplitudeRamp_ms = 2; % ramp (envelope) for sound onset/offset
    % IF USING AMPLITUDE RAMP AS PARAM, THEN NEEDS TO BE USED TO CONSTRUCT
    % AUDIO ENVELOPE IN CODE BELOW
    %S.GUIPanels.Sound = {'DigitalAttenuation_dB', 'AmplitudeRamp_ms'};
    S.GUIPanels.Sound = {'DigitalAttenuation_dB'};

    %% init cue params
    S.GUI.InitCueDuration_s = 0.05; % Duration of init sound
    S.GUI.InitWindowTimeout_s = 5; % How long the mouse has to initiate stimulus or miss init lick
    S.GUI.InitCueFreq_Hz = 500; % Frequency of init cue
    S.GUIPanels.InitCue = {'InitCueDuration_s', 'InitWindowTimeout_s', 'InitCueFreq_Hz'};

    %% go cue params
    S.GUI.GoCueDuration_s = 0.05; % Duration of go sound
    S.GUI.GoCueFreq_Hz = 2000; % Frequency of go cue
    S.GUIPanels.GoCue = {'GoCueDuration_s', 'GoCueFreq_Hz'};

    %% training level params
    S.GUI.TrainingLevel = 1; % Configurable training and test schemes.
                             % 1 - 'Naive', 2 - 'Trained'
    S.GUIMeta.TrainingLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
    S.GUIMeta.TrainingLevel.String = {'Naive', 'Well Trained'};
    S.GUI.NumEasyWarmupTrials = 20;
    S.GUIPanels.Training = {'TrainingLevel', 'NumEasyWarmupTrials'};

    %% difficulty params
    % percentage of full perturbation range as boundaries for difficulty levels
    S.GUI.PercentTrialsEasy = 100;
    S.GUI.PercentTrialsMediumEasy = 0;
    S.GUI.PercentTrialsMediumHard = 0;
    S.GUI.PercentTrialsHard = 0;
    %S.GUIPanels.Difficulty = {'EasyMinPercent', 'EasyMaxPercent', 'MediumMinPercent', 'MediumMaxPercent', 'HardMinPercent', 'HardMaxPercent'};
    %S.GUIPanels.Difficulty = {'PercentTrialsEasy', 'PercentTrialsMedium', 'PercentTrialsHard', 'EasyMinPercent', 'EasyMaxPercent', 'MediumMinPercent', 'MediumMaxPercent', 'HardMinPercent', 'HardMaxPercent'};
    %S.GUIPanels.Difficulty = {'DifficultyType', 'PercentTrialsEasy', 'PercentTrialsMedium', 'PercentTrialsHard', 'Difficulty'};
    S.GUIPanels.Difficulty = {'PercentTrialsEasy', 'PercentTrialsMediumEasy', 'PercentTrialsMediumHard', 'PercentTrialsHard'};

    %% vis stim params
    %S.GUI.GratingDur_s = 0.25; % Duration of grating stimulus in seconds - ORIGINAL
    S.GUI.GratingDur_s = 0.1; % Duration of grating stimulus in seconds - UPDATE
    %S.GUI.ISIOrig_s = 0.75; % Duration of *fixed* gray screen stimulus in seconds - ORIGINAL
    S.GUI.ISIOrig_s = 0.5; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
    S.GUI.ExtraStimDurPostRew_Naive_s = 5; % naive mouse sees stimulus for this time (sec) after correct lick    
    S.GUI.NumISIOrigRep = 5; % number of grating/gray repetitions for vis stim first segment prior to perturbation
    S.GUI.PostPerturbDurMultiplier = 1.5; % scaling factor for post perturbation stimulus (postperturb = preperturb * PostPerturbDurMultiplier)
    %S.GUI.NumPerturbVisRep = 5;  % number of gray/grating repetitions for vis stim second segment (perturbation) associated with L/R choice    
    %S.GUI.PerturbMinFromCB_ms = 0; % min time in ms for perturbation range from category boundary
    S.GUI.MinISIPerturb_ms = 100; % min time in ms for perturbation range from grating
    S.GUI.PreVisStimDelay_s = 0; % How long the mouse must poke in the center to activate the goal port
    S.GUI.PreGoCueDelay_s = 0;

    S.GUIPanels.VisStim = {'GratingDur_s', 'ISIOrig_s', 'ExtraStimDurPostRew_Naive_s', 'NumISIOrigRep', 'PostPerturbDurMultiplier', 'MinISIPerturb_ms', 'PreVisStimDelay_s', 'PreGoCueDelay_s'}; 
    %S.GUIPanels.VisStim = {'GratingDuration', 'ISI', 'ExtraStimDurPostRew_Naive', 'PreVisStimDelay', 'PreGoCueDelay', 'NumISIOrigRep', 'NumPerturbVisRep', 'PerturbMinFromCB_ms', 'PerturbMinFromGrating_ms'}; 
    % 3 - 'PrePertVisStim',
    % if debug
    %     S.GUI.GratingDur_s = 1;
    %     S.GUI.ISIOrig_s = 1;
    % end   

    %% contingency and bias params
    S.GUI.ShortISIChoice = 1;   % set short ISI association to left or right side
                                % 1 - 'Left', 2 - 'Right'
    S.GUIMeta.ShortISIChoice.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
    S.GUIMeta.ShortISIChoice.String = {'Left', 'Right'};

    S.GUI.ShortISIFraction = 0.5;   % set fraction of trials that are short ISI (long ISI fraction = (1 - short))
                                % 1 - 'Left', 2 - 'Right'

    S.GUI.ManualSideSelect = 0;   % override to disable/enable manual selection of left/right for next trial
    S.GUIMeta.ManualSideSelect.Style = 'checkbox';
    S.GUI.ManualSide = 1;   % manual selection of left/right for next trial
    S.GUIMeta.ManualSide.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
    S.GUIMeta.ManualSide.String = {'Left', 'Right'};

    S.GUIPanels.Contingency_Bias = {'ShortISIChoice', 'ShortISIFraction', 'ManualSideSelect', 'ManualSide'};

    %% reward params
    S.GUI.LeftValveTime_s = 0.25;
    S.GUI.RightValveTime_s = 0.25;
    S.GUI.CenterValveTime_s = 0.10;
    S.GUI.WindowRewardGrabDuration_Naive_s = 10;  % naive mouse has up to x seconds to grab reward    
    S.GUI.RewardDelay_s = 0; % How long the mouse must wait in the goal port for reward to be delivered
   
    S.GUIPanels.Reward = {'LeftValveTime_s', 'RightValveTime_s', 'CenterValveTime_s', 'WindowRewardGrabDuration_Naive_s', 'RewardDelay_s'};

    %% punish params
    S.GUI.PunishSoundDuration_s = 2; % Seconds to wait on errors before next trial can start
    S.GUI.IncorrectSound = 1; % if 1, plays a white noise pulse on error. if 0, no sound is played.
    S.GUIMeta.IncorrectSound.Style = 'checkbox';
    S.GUIPanels.Punish = {'PunishSoundDuration_s', 'IncorrectSound'}; 

    %% choice params
    % choice parameters
    S.GUI.ChoiceWindow_s = 5; % How long after go cue until the mouse must make a choice
    S.GUI.ConfirmLickInterval_s = 0.2; % min interval until choice can be confirmed    
    S.GUI.ChoiceConfirmWindow_s = 5; % time during which correct choice can be confirmed
    %S.GUI.ChoiceReattemptWindow = 2; % time during which choice can be re-attempted after not confirming correct/incorrect choice
    %S.GUIPanels.Choice = {'ChoiceWindow', 'ChoiceReattemptWindow', 'ConfirmLickInterval', 'ChoiceConfirmWindow'};
    S.GUIPanels.Choice = {'ChoiceWindow_s', 'ConfirmLickInterval_s', 'ChoiceConfirmWindow_s'};  


    S.GUI.RewardAmount = 3; %Unit = ?l
    S.GUI.InitDelay = 1; % How long the test subject must keep the wheel motionless to receive the stimulus. Unit = seconds
    S.GUI.ResponseTime = 5; % How long until the subject must make a choice, or forefeit the trial
    S.GUI.ErrorDelay = 3; % How long the subject must wait to start the next trial after an incorrect choice
    S.GUI.InitThreshold = 3; % How much the wheel may move during the initialization period without resetting the init delay. Units = degrees
    S.GUI.ChoiceThreshold = 40; % Wheel position in the correct or incorrect direction at which a choice is registered. Units = degrees (unsigned)
    S.GUI.GaborTilt = 0; % Degrees
    S.GUI.GaborContrast = 100; % Percent of full scale
end

%% Display any needed informtion at start of session
disp('Running Session Setup');

%% Define trials
MaxTrials = 1000;
TrialTypes = ceil(rand(1,MaxTrials)*2);

% project future reward plot according to gui param default
% switch S.GUI.RewardedSideForCase1
%     case 1 % left side rewarded for case 1
%         TrialSidesRewarded = TrialTypes;
%     case 2 % right side rewarded for case 1
%         TrialSidesRewarded = (3-TrialTypes);
% end

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


%% Define trials
MaxTrials = 1000; % Maximum number of trials in the session. Session can be manually ended before MaxTrials from the console GUI.
TrialTypes = round(rand(1,MaxTrials)); % TrialType 0 = left correct, TrialType 1 = rightCorrect
CorrectDirection = TrialTypes; CorrectDirection(CorrectDirection == 0) = -1; % Correct response direction for each trial (-1 for left, 1 for right)
BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.


%% Initialize plots
% Side Outcome Plot
BpodSystem.ProtocolFigures.SideOutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
%BpodSystem.GUIHandles.SideOutcomePlot = axes('Position', [.075 .35 .89 .55]);
BpodSystem.GUIHandles.SideOutcomePlot = axes('Position', [.08 .3 .89 .6]);
%SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'init',2-TrialTypes);
SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'init',2-TrialTypes);
%TotalRewardDisplay('init'); % Total Reward display (online display of the total amount of liquid reward earned)
%BpodNotebook('init');
BpodParameterGUI('init', S); % Initialize parameter GUI plugin

%% Setup rotary encoder module
R.useAdvancedThresholds = 'on'; % Advanced thresholds are available on rotary encoder module r2.0 or newer.
                                % See notes in setAdvancedThresholds() function in /Modules/RotaryEncoderModule.m for parameters and usage
R.sendThresholdEvents = 'on'; % Enable sending threshold crossing events to state machine
R.userCallbackFcn = 'WheelPosCallback';
R.setAdvancedThresholds([-S.GUI.ChoiceThreshold S.GUI.ChoiceThreshold S.GUI.InitThreshold], [0 0 1], [0 0 S.GUI.InitDelay]); % Syntax: setAdvancedThresholds(thresholds, thresholdTypes, thresholdTimes)

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
SF = 192000; % Use max supported sampling rate
H.SamplingRate = SF;
InitCueSound = GenerateSineWave(SF, S.GUI.InitCueFreq_Hz, S.GUI.InitCueDuration_s)*.9; % Sampling freq (hz), Sine frequency (hz), duration (s)
GoCueSound = GenerateSineWave(SF, S.GUI.GoCueFreq_Hz, S.GUI.GoCueDuration_s)*.9; % Sampling freq (hz), Sine frequency (hz), duration (s)
IncorrectSound = GenerateWhiteNoise(SF, S.GUI.PunishSoundDuration_s, 1, 2); % white noise punish sound

%H.HeadphoneAmpEnabled = true; H.HeadphoneAmpGain = 15; % Ignored if using HD version of the HiFi module
H.DigitalAttenuation_dB = -30; % Set a comfortable listening level for most headphones (useful during protocol dev).
H.load(1, InitCueSound);
H.load(2, GoCueSound);
H.load(3, IncorrectSound);
Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset
H.AMenvelope = Envelope;

% Remember values of sound frequencies & durations, so a new one only gets uploaded if it was changed
LastInitCueFrequency = S.GUI.InitCueFreq_Hz; 
LastGoCueFrequency = S.GUI.GoCueFreq_Hz;

LastInitCueDuration = S.GUI.InitCueDuration_s;
LastGoCueDuration = S.GUI.GoCueDuration_s;
LastPunishSoundDuration = S.GUI.PunishSoundDuration_s;

%LastSoundDuration = S.GUI.SoundDuration;

%% Setup video
if isfield(BpodSystem.PluginObjects, 'V') % Clear previous instances of the video server
    BpodSystem.PluginObjects.V = [];
end
MonitorID = 2;
BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(MonitorID, 0, [0 0], [180 180], 0); % Assumes second monitor is screen #2. Sync patch = 180x180 pixels
BpodSystem.PluginObjects.V.SyncPatchIntensity = 140;
% Indicate loading
BpodSystem.PluginObjects.V.loadText(1, 'Loading...', '', 80);
BpodSystem.PluginObjects.V.play(1);

Xsize = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
Ysize = BpodSystem.PluginObjects.V.ViewPortDimensions(1);

% compute grating according to square grid of largest pixel dimension
if Xsize > Ysize
    gratingSize = [Xsize, Xsize]; % Size of grating in pixels
else
    gratingSize = [Ysize, Ysize]; % Size of grating in pixels
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

gray = gray(1:Xsize, 1:Ysize); % clip to monitor
sinGrating = sinGrating(1:Xsize, 1:Ysize); % clip to monitor

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

GratingPattern = [GratingFrame_SyncW GratingFrame_SyncBlk];
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

GrayPattern = [GrayFrame_SyncW GrayFrame_SyncBlk];
%GrayVideo = [repmat(GrayPattern, 1, GrayFixedFrames/2) GrayBlank];
GrayVideo = [repmat(GrayPattern, 1, GrayFixedFrames/2)];

BpodSystem.PluginObjects.V.Videos{4} = struct;
BpodSystem.PluginObjects.V.Videos{4}.nFrames = GrayFixedFrames + 1; % + 1 for final frame
BpodSystem.PluginObjects.V.Videos{4}.Data = [GrayVideo GrayBlank];

%VideoInitialBasePattern = [GratingVideo GrayVideo]; % base video pattern for initial segment repetitions of grating->gray
%VideoInitialBasePattern = [GratingVideo]; % base video pattern for initial segment repetitions of grating->gray

%BaseVideo = [];

%NumISIOrigRep = S.GUI.NumISIOrigRep;

%VideoData = [repmat(VideoInitialBasePattern, 1, S.GUI.NumISIOrigRep) GratingVideo]; % add one more grating video after initial repetitions
%VideoData = GratingVideo; % for joystick proto 0, only single grating is used

%PrePerturbDur = (GratingDuration + GrayFixedDuration) * S.GUI.NumISIOrigRep + GratingDuration;

% if ShowDebugOutput
%     disp(['PrePerturbDur: ', num2str(PrePerturbDur)]);
% end

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
%LastGrayFixedDuration = S.GUI.ISIOrig_s; % Remember value of pre-perturb gray dur so that we only regenerate the pre-perturb gray video if parameter has changed
%LastNumISIOrigRep = S.GUI.NumISIOrigRep;  % Remember value of initial segment repetitions so that we only regenerate the initial segment if parameter changed

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
BpodSystem.PluginObjects.V.TimerMode = 1;
pause(1.0); % matlab seems to require a pause here before clearing screen with play(0), 
            % otherwise can get stuck on Psychtoolbox splash screen
            % might need longer delay if purple image hangs on window open
BpodSystem.PluginObjects.V.play(0);
BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_PlayVideo';

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

%% stream encoder position data
R.startUSBStream; % Begin streaming position data to PC via USB

%% Main trial loop
for currentTrial = 1:MaxTrials
    S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin
    %ValveTime = GetValveTimes(S.GUI.RewardAmount, 1); % Update reward amount
    % S.GUI.LeftValveTime_s = 0.25;
    % S.GUI.RightValveTime_s = 0.25;
    % S.GUI.CenterValveTime_s = 0.10;
    %R.setAdvancedThresholds([-S.GUI.ChoiceThreshold S.GUI.ChoiceThreshold S.GUI.InitThreshold], [0 0 1], [0 0 S.GUI.InitDelay]); % Syntax: setAdvancedThresholds(thresholds, thresholdTypes, thresholdTimes)
    R.setAdvancedThresholds([-S.GUI.ChoiceThreshold S.GUI.ChoiceThreshold S.GUI.InitThreshold 0], [0 0 1 0], [0 0 S.GUI.InitDelay 0]); % Syntax: setAdvancedThresholds(thresholds, thresholdTypes, thresholdTimes)
    % added forth thresh for detect returning joystick to zero

    switch TrialTypes(currentTrial) % Determine trial-specific state matrix fields
        case 0
            LeftChoiceAction = 'LeftReward'; RightChoiceAction = 'Error';
            ValveTime = S.GUI.LeftValveTime_s;
        case 1
            LeftChoiceAction = 'Error'; RightChoiceAction = 'RightReward'; 
            ValveTime = S.GUI.RightValveTime_s;
    end

    switch S.GUI.TrainingLevel
        case 1 % naive        
            if ShowDebugOutput
                disp('Training Stage - Naive');
            end
            %InitCue_Tup_NextState = 'InitReward'; % change to fn_lic_poiss_3      
            InitCue_Tup_NextState = 'InitWindow';
            %DidNotChoose_Tup_NextState = 'Reward';
            DidNotChoose_Tup_NextState = 'ITI';
    
            VisualStimulusStateChangeConditions = {'Tup', 'PreGoCueDelay'};
    
            %PreGoCueDelay_OutputActions = {}; % no output action in PreGoCueDelay for naive
            CenterReward_OutputActions = {'Valve2', 1};

            %GoCue_Tup_NextState = 'RewardNaive';  % naive
            GoCue_Tup_NextState = 'WindowChoice';  % trained

            PunishSetup_Tup_NextState = 'PunishNaive'; % Naive
        
            ExperimenterTrialInfo.TrainingLevel = 'Naive';
        case 2 % well trained
            if ShowDebugOutput
                disp('Training Stage - Trained');
            end
            InitCue_Tup_NextState = 'InitWindow';
            DidNotChoose_Tup_NextState = 'ITI';
    
            %VisualStimulusStateChangeConditions = {'Tup', 'CenterReward', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
            VisualStimulusStateChangeConditions = {'Tup', 'CenterReward', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
    
           % PreGoCueDelay_OutputActions = {'SoftCode', 255}; % stop vis stim in PreGoCueDelay so its init and perturb segment durations are equal for well trained
            CenterReward_OutputActions = {'Valve2', 1, 'SoftCode', 255}; % moved video stop code earlier to center reward

            GoCue_Tup_NextState = 'WindowChoice';  % trained
            PunishSetup_Tup_NextState = 'Punish'; % trained

            ExperimenterTrialInfo.TrainingLevel = 'Well Trained';
    end

    %% trial-specific output arguments for sounds
    % maybe move this to state-matrix section for consistency
    OutputActionArgInitCue = {'HiFi1', ['P' 0], 'BNCState', 1};
    OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};

    Valve = 'Valve2';

    sma = NewStateMachine(); % Assemble new state machine description
    %sma = NewStateMatrix(); % Assemble state matrix, which version should
    %this be?
    %sma = SetCondition(sma, 1, 'Port1', 0); % Condition 1: Port 1 low (is out)    
    sma = SetCondition(sma, 1, 'Port2', 1); % Condition 1:  is (licking) 
    
    sma = AddState(sma, 'Name', 'Start', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ZeroEncoder'},...
        'OutputActions', {'RotaryEncoder1', ['#' 0]}); 
    % 'RotaryEncoder1' '#' marks a trial start timestamp in the rotary encoder data stream (for sync)
    % See https://sites.google.com/site/bpoddocumentation/user-guide/serial-interfaces for a list of all byte commands      
    sma = AddState(sma, 'Name', 'ZeroEncoder', ... 
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'InitCue'},...
        'OutputActions', {'RotaryEncoder1', '*Z'}); % '*' = push new thresholds to rotary encoder 'Z' = zero position  
    % sma = AddState(sma, 'Name', 'InitDelay', ... % Turn on LED of port1. Wait for InitDelay seconds. Ensure that wheel does not move.
    %     'Timer', 0,...
    %     'StateChangeConditions', {'RotaryEncoder1_3', 'DeliverStimulus', 'RotaryEncoder1_4', 'DeliverStimulus'},...
    %     'OutputActions', {'LED', 1, 'RotaryEncoder1', [';' 4]}); % ';' = enable thresholds specified by bits of a byte. 4 = binary 100 (enable threshold# 3)                                      
    sma = AddState(sma, 'Name', 'InitCue', ...
        'Timer', S.GUI.InitCueDuration_s,...
        'StateChangeConditions', {'Tup', InitCue_Tup_NextState},...         
        'OutputActions', OutputActionArgInitCue);
    %'OutputActions', {'LED', 1, 'RotaryEncoder1', [';' 4]}); % ';' = enable thresholds specified by bits of a byte. 4 = binary 100 (enable threshold# 3)                                      
 
    % naive - center spout reward to condition init licking
    % maybe give this it's own unique valve time?
    % sma = AddState(sma, 'Name', 'InitReward', ...
    %     'Timer', CenterValveTime,...
    %     'StateChangeConditions', {'Tup', 'InitWindow'},...
    %     'OutputActions', {'Valve2', 1});    

    % outdated comment, update
    % naive - auto go to vis stimulus after playing go cue sound (we still
    %         log DidNotInitiate, but go to vis stim instead of ITI)
    % trained - wait for lick after playing go cue sound
    sma = AddState(sma, 'Name', 'InitWindow', ...
        'Timer', S.GUI.InitWindowTimeout_s,...
        'StateChangeConditions', {'Tup', 'InitCueAgain', 'Port2In', 'ConfirmInit'},...
        'OutputActions', {});    
    %'StateChangeConditions', {'Tup', 'InitCueAgain', 'Port2In', 'ConfirmInit', 'Condition1', 'ConfirmInit'},...  % maybe this one in case already holding handle/spout/whatever  
    %'StateChangeConditions', {'Tup', 'InitCueAgain', 'Port2In', 'PreVisStimDelay', 'Port1In', 'WrongInitiation', 'Port3In', 'WrongInitiation'},...         

    sma = AddState(sma, 'Name', 'InitCueAgain', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'InitCue'},...         
        'OutputActions', {});  

    sma = AddState(sma, 'Name', 'ConfirmInit', ...
        'Timer', 0,...                  % can change timing here to require 'hold' for confirm
        'StateChangeConditions', {'Tup', 'VisualStimulus1', 'Port2Out', 'InitCueAgain'},...         
        'OutputActions', {});
%% first rep

    % VisualStimulus
    sma = AddState(sma, 'Name', 'VisualStimulus1', ...
        'Timer', S.GUI.GratingDur_s,...
        'StateChangeConditions', {'Tup', 'GoCue1'},...
        'OutputActions', {'SoftCode', 3, 'RotaryEncoder1', [';' 3]});
        %'StateChangeConditions', VisualStimulusStateChangeConditions,...

    % play go cue sound
    sma = AddState(sma, 'Name', 'GoCue1', ...
        'Timer', S.GUI.GoCueDuration_s,...
        'StateChangeConditions', {'Tup', 'WindowChoice1'},...         
        'OutputActions', OutputActionArgGoCue);
    %'StateChangeConditions', {'Tup', GoCue_Tup_NextState},...  

    % naive - don't require confirm/choose again loop, log DidNotChoose,
    %         reward from correct spout anyway
    % trained - 
    % well trained - ?
    sma = AddState(sma, 'Name', 'WindowChoice1', ...
        'Timer', S.GUI.ChoiceWindow_s,...
        'StateChangeConditions', {'Tup', 'DidNotChoose', 'RotaryEncoder1_1', 'ResetJoystickPos1'},...
        'OutputActions', {});      
        %'StateChangeConditions', {CorrectLick, 'CorrectLickInterval', IncorrectLick, 'IncorrectLickInterval', 'Tup', 'DidNotChoose'},...
    
    % return to start pos using stepper motor goes here

    % detect joystick has returned to center
    sma = AddState(sma, 'Name', 'ResetJoystickPos1', ...
        'Timer', 0,...
        'StateChangeConditions', {'RotaryEncoder1_4', 'VisualStimulus2'},...
        'OutputActions', {'RotaryEncoder1', [';' 8]}); 
        %'OutputActions', {'RotaryEncoder1', [';' 3]});

%% second rep
    % VisualStimulus
    sma = AddState(sma, 'Name', 'VisualStimulus2', ...
        'Timer', S.GUI.GratingDur_s,...
        'StateChangeConditions', {'Tup', 'WindowChoice2'},...
        'OutputActions', {'SoftCode', 3, 'RotaryEncoder1', [';' 3]});  % rest threshold for push
        %'StateChangeConditions', VisualStimulusStateChangeConditions,...
 
    % naive - don't require confirm/choose again loop, log DidNotChoose,
    %         reward from correct spout anyway
    % trained - 
    % well trained - ?
    sma = AddState(sma, 'Name', 'WindowChoice2', ...
        'Timer', S.GUI.ChoiceWindow_s,...
        'StateChangeConditions', {'Tup', 'DidNotChoose', 'RotaryEncoder1_1', 'ResetJoystickPos2'},...
        'OutputActions', {});      
        %'StateChangeConditions', {CorrectLick, 'CorrectLickInterval', IncorrectLick, 'IncorrectLickInterval', 'Tup', 'DidNotChoose'},...
    
    % return to start pos using stepper motor goes here

    % detect joystick has returned to center
    sma = AddState(sma, 'Name', 'ResetJoystickPos2', ...
        'Timer', 0,...
        'StateChangeConditions', {'RotaryEncoder1_4', 'VisualStimulus3'},...
        'OutputActions', {'RotaryEncoder1', [';' 8]}); % reset forth threshold '1000' for detect return to center
        %'OutputActions', {'RotaryEncoder1', [';' 3]});

%% third rep
    % VisualStimulus
    sma = AddState(sma, 'Name', 'VisualStimulus3', ...
        'Timer', S.GUI.GratingDur_s,...
        'StateChangeConditions', {'Tup', 'WindowChoice3'},...
        'OutputActions', {'SoftCode', 3, 'RotaryEncoder1', [';' 3]});  % rest threshold for push
        %'StateChangeConditions', VisualStimulusStateChangeConditions,...
 
    % naive - don't require confirm/choose again loop, log DidNotChoose,
    %         reward from correct spout anyway
    % trained - 
    % well trained - ?
    sma = AddState(sma, 'Name', 'WindowChoice3', ...
        'Timer', S.GUI.ChoiceWindow_s,...
        'StateChangeConditions', {'Tup', 'DidNotChoose', 'RotaryEncoder1_1', 'ResetJoystickPos3'},...
        'OutputActions', {});      
        %'StateChangeConditions', {CorrectLick, 'CorrectLickInterval', IncorrectLick, 'IncorrectLickInterval', 'Tup', 'DidNotChoose'},...
    
    % return to start pos using stepper motor goes here

    % detect joystick has returned to center
    sma = AddState(sma, 'Name', 'ResetJoystickPos3', ...
        'Timer', 0,...
        'StateChangeConditions', {'RotaryEncoder1_4', 'VisualStimulus4'},...
        'OutputActions', {'RotaryEncoder1', [';' 8]}); % reset forth threshold '1000' for detect return to center
        %'OutputActions', {'RotaryEncoder1', [';' 3]});    

%% forth rep
    % VisualStimulus
    sma = AddState(sma, 'Name', 'VisualStimulus4', ...
        'Timer', S.GUI.GratingDur_s,...
        'StateChangeConditions', {'Tup', 'WindowChoice4'},...
        'OutputActions', {'SoftCode', 3, 'RotaryEncoder1', [';' 3]});  % rest threshold for push
        %'StateChangeConditions', VisualStimulusStateChangeConditions,...
 
    % naive - don't require confirm/choose again loop, log DidNotChoose,
    %         reward from correct spout anyway
    % trained - 
    % well trained - ?
    sma = AddState(sma, 'Name', 'WindowChoice4', ...
        'Timer', S.GUI.ChoiceWindow_s,...
        'StateChangeConditions', {'Tup', 'DidNotChoose', 'RotaryEncoder1_1', 'ResetJoystickPos4'},...
        'OutputActions', {});      
        %'StateChangeConditions', {CorrectLick, 'CorrectLickInterval', IncorrectLick, 'IncorrectLickInterval', 'Tup', 'DidNotChoose'},...
    
    % return to start pos using stepper motor goes here

    % detect joystick has returned to center
    sma = AddState(sma, 'Name', 'ResetJoystickPos4', ...
        'Timer', 0,...
        'StateChangeConditions', {'RotaryEncoder1_4', 'VisualStimulus5'},...
        'OutputActions', {'RotaryEncoder1', [';' 8]}); % reset forth threshold '1000' for detect return to center
        %'OutputActions', {'RotaryEncoder1', [';' 3]});   

%% fifth rep
    % VisualStimulus
    sma = AddState(sma, 'Name', 'VisualStimulus5', ...
        'Timer', S.GUI.GratingDur_s,...
        'StateChangeConditions', {'Tup', 'WindowChoice5'},...
        'OutputActions', {'SoftCode', 3, 'RotaryEncoder1', [';' 3]});  % rest threshold for push
        %'StateChangeConditions', VisualStimulusStateChangeConditions,...
 
    % naive - don't require confirm/choose again loop, log DidNotChoose,
    %         reward from correct spout anyway
    % trained - 
    % well trained - ?
    sma = AddState(sma, 'Name', 'WindowChoice5', ...
        'Timer', S.GUI.ChoiceWindow_s,...
        'StateChangeConditions', {'Tup', 'DidNotChoose', 'RotaryEncoder1_1', 'ResetJoystickPos5'},...
        'OutputActions', {});      
        %'StateChangeConditions', {CorrectLick, 'CorrectLickInterval', IncorrectLick, 'IncorrectLickInterval', 'Tup', 'DidNotChoose'},...
    
    % return to start pos using stepper motor goes here

    % detect joystick has returned to center
    sma = AddState(sma, 'Name', 'ResetJoystickPos5', ...
        'Timer', 0,...
        'StateChangeConditions', {'RotaryEncoder1_4', 'Reward'},...
        'OutputActions', {'RotaryEncoder1', [';' 8]}); % reset forth threshold '1000' for detect return to center
        %'OutputActions', {'RotaryEncoder1', [';' 3]});         

%% post task

    sma = AddState(sma, 'Name', 'Reward', ...
        'Timer', ValveTime,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {Valve, 1});


    sma = AddState(sma, 'Name', 'DidNotChoose', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});


    sma = AddState(sma, 'Name', 'DeliverStimulus', ...
        'Timer', S.GUI.ResponseTime,...
        'StateChangeConditions', {'RotaryEncoder1_1', LeftChoiceAction, 'RotaryEncoder1_2', RightChoiceAction, 'Tup', 'TimedOut'},...
        'OutputActions', {'RotaryEncoder1', ['Z;' 3]}); 
    sma = AddState(sma, 'Name', 'LeftReward', ...
        'Timer', ValveTime,...
        'StateChangeConditions', {'Tup', 'Drinking'},...
        'OutputActions', {'Valve1', 1}); % Sound 1 is reward feedback. Valve 1 is open during this state.
    sma = AddState(sma, 'Name', 'RightReward', ...
        'Timer', ValveTime,...
        'StateChangeConditions', {'Tup', 'Drinking'},...
        'OutputActions', {'Valve1', 1});
    sma = AddState(sma, 'Name', 'Drinking', ...
        'Timer', 0,...
        'StateChangeConditions', {'Condition1', 'DrinkingGrace'},...
        'OutputActions', {});
    sma = AddState(sma, 'Name', 'DrinkingGrace', ...
        'Timer', 0.5,...
        'StateChangeConditions', {'Tup', 'ITI', 'Port1In', 'Drinking'},...
        'OutputActions', {});
    sma = AddState(sma, 'Name', 'Error', ...
        'Timer', S.GUI.ErrorDelay,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});
    sma = AddState(sma, 'Name', 'TimedOut', ...
        'Timer', S.GUI.ErrorDelay,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {}); % 'X' cancels any ongoing sound on the HiFi module

        sma = AddState(sma, 'Name', 'ITI', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', '>exit'},...
        'OutputActions', {'SoftCode', 255});  
        %'Timer', ITI,...

    SendStateMachine(sma);
    RawEvents = RunStateMachine;
    if ~isempty(fieldnames(RawEvents)) % If trial data was returned
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
        BpodSystem.Data.CorrectDirection(currentTrial) = CorrectDirection(currentTrial);
        BpodSystem.Data.EncoderData{currentTrial} = R.readUSBStream(); % Get rotary encoder data captured since last call to R.readUSBStream()
        BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
        BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial); % Adds the trial type of the current trial to data
        UpdateSideOutcomePlot(TrialTypes, BpodSystem.Data);
        % Align this trial's rotary encoder timestamps to state machine trial-start (timestamp of '#' command sent from state machine to encoder module in 'Start' state)
        BpodSystem.Data.EncoderData{currentTrial}.Times = BpodSystem.Data.EncoderData{currentTrial}.Times - BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps(1); % Align timestamps to state machine's trial time 0
        BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps = BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps - BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps(1); % Align event timestamps to state machine's trial time 0
        SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file
    end
    HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
    if BpodSystem.Status.BeingUsed == 0
        R.stopUSBStream; % Stop streaming positions from rotary encoder module
        clear R;
        %BpodSystem.PluginObjects.Gabor = [];
        BpodSystem.PluginObjects.V = [];
        return
    end
end
R.stopUSBStream;
%BpodSystem.PluginObjects.Gabor = [];
BpodSystem.PluginObjects.V = [];

function UpdateSideOutcomePlot(TrialTypes, Data) 
global BpodSystem
Outcomes = zeros(1,Data.nTrials);
for x = 1:Data.nTrials % Encode user data for side outcome plot plugin
    if ~isnan(Data.RawEvents.Trial{x}.States.Drinking(1))
        Outcomes(x) = 1;
    elseif ~isnan(Data.RawEvents.Trial{x}.States.Error(1))
        Outcomes(x) = 0;
    elseif ~isnan(Data.RawEvents.Trial{x}.States.TimedOut(1))
        Outcomes(x) = 2;
    else
        Outcomes(x) = 3;
    end
end
SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'update',Data.nTrials+1,TrialTypes,Outcomes);