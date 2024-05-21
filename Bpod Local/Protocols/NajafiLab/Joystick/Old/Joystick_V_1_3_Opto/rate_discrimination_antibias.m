function rate_discrimination_antibias

global BpodSystem

%% Import scripts

m_Plotter = Plotter;
m_InitGUI = InitGUI;
m_TrialConfig = TrialConfig;


%% Turn off Bpod LEDs

% This code will disable the state machine status LED
BpodSystem.setStatusLED(0);

%% Assert HiFi module is present + USB-paired (via USB button on console GUI)

BpodSystem.assertModule('HiFi', 1); % The second argument (1) indicates that the HiFi module must be paired with its USB serial port
% Create an instance of the HiFi module
H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1); % The argument is the name of the HiFi module's USB serial port (e.g. COM3)

%% Define parameters

[S] = m_InitGUI.SetParams(BpodSystem);


%% Define trials

% set max number of trials
MaxTrials = 1000;

% initialize anti-bias variables
AntiBiasVar.IncorrectFlag = 0;
AntiBiasVar.IncorrectType = 1;
AntiBiasVar.CompletedHist.left = [];
AntiBiasVar.CompletedHist.right = [];
AntiBiasVar.ValveFlag = 0;

% get uniform distribution of 2 trial types
TrialTypes = ceil(rand(1,MaxTrials)*2); 

% adjust warmup trials to have no more than 'max' number of consecutive
% same-side trials
[TrialTypes] = m_TrialConfig.AdjustWarmupTrials(S, TrialTypes);
   
BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.



%% Initialize plots

% Side Outcome Plot
BpodSystem.ProtocolFigures.SideOutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.SideOutcomePlot = axes('Position', [.075 .35 .89 .55]);
SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'init',2-TrialTypes);
BpodParameterGUI('init', S); % Initialize parameter GUI plugin

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
H.load(1, InitCueSound);
H.load(2, GoCueSound);
H.load(3, IncorrectSound);


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

VideoPrePerturbPattern = [GratingVideo GrayVideo]; % base video pattern for initial segment repetitions of grating->gray

% try to get Frame2TTL BNC sync to be low and not miss first frame of vis
% stim later
GrayInitBNCSync = [repmat(GrayFrame_SyncW, 1, 120) GrayFrame_SyncBlk];
BpodSystem.PluginObjects.V.Videos{6} = struct;
BpodSystem.PluginObjects.V.Videos{6}.nFrames = 121; % + 1 for final frame
BpodSystem.PluginObjects.V.Videos{6}.Data = GrayInitBNCSync;

VideoData = [repmat(VideoPrePerturbPattern, 1, S.GUI.NumISIOrigRep) GratingVideo]; % add one more grating video after initial repetitions

PrePerturbDur = (GratingDuration + GrayFixedDuration) * S.GUI.NumISIOrigRep + GratingDuration;

LastGratingDuration = S.GUI.GratingDur_s; % Remember value of stim dur so that we only regenerate the grating video if parameter has changed
LastGrayFixedDuration = S.GUI.ISIOrig_s; % Remember value of pre-perturb gray dur so that we only regenerate the pre-perturb gray video if parameter has changed
LastNumISIOrigRep = S.GUI.NumISIOrigRep;  % Remember value of initial segment repetitions so that we only regenerate the initial segment if parameter changed

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

% define discrete values in distribution
DifficultyLevels = [1, 2, 3, 4];  % 1 - Easy, 2 - MediumEasy, 3 - MediumHard, 4 - Hard

WarmupTrialsCounter = S.GUI.NumEasyWarmupTrials;
LastNumEasyWarmupTrials = S.GUI.NumEasyWarmupTrials; % store GUI value to determine if user has changed this param to reset counter


%% 
LastWaitDurOrig_s = S.GUI.WaitDurOrig_s;
LastWaitDurStep_s = S.GUI.WaitDurStep_s;
% init wait dur
wait_dur = 0;

%% Main trial loop

for currentTrial = 1:MaxTrials
   
    ExperimenterTrialInfo.TrialNumber = currentTrial;   % capture variable states as field/value struct for experimenter info

    %% sync trial-specific parameters from GUI

    S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin    


    %% anti bias

    [TrialTypes] = m_TrialConfig.ManuallFraction( ...
        S, currentTrial, TrialTypes);

    [TrialTypes, AntiBiasVar] = m_TrialConfig.RepeatedIncorrect( ...
        BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes);

    [AntiBiasVar, LeftValveTime, RightValveTime, TrialTypes] = m_TrialConfig.AntiBiasValveAdjust( ...
        BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes);


    %% update trial-specific valve times according to set reward amount

    CenterValveTime = S.GUI.CenterValveTime_s;
    ExperimenterTrialInfo.LeftValveTime = LeftValveTime;
    ExperimenterTrialInfo.RightValveTime = RightValveTime;


    %% set state matrix vars according to contigency

    switch TrialTypes(currentTrial) % Determine trial-specific state matrix fields
        case 1 % left side is rewarded
            CorrectLick = 'Port1In';
            CorrectPort = 'Port1';
            IncorrectLick = 'Port3In';
            IncorrectPort = 'Port3';
            ValveTime = LeftValveTime; 
            Valve = 'Valve1';
        case 2  % right side is rewarded 
            CorrectLick = 'Port3In';
            CorrectPort = 'Port3';
            IncorrectLick = 'Port1In';
            IncorrectPort = 'Port1';
            ValveTime = RightValveTime; 
            Valve = 'Valve3';
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
        ExperimenterTrialInfo.Warmup = true;   % capture variable states as field/value struct for experimenter info
        ExperimenterTrialInfo.WarmupTrialsRemaining = WarmupTrialsCounter;   % capture variable states as field/value struct for experimenter info

        TrialDifficulty = 1;  % set warmup trial to easy     
    else 

        PercentTrialsEasy = S.GUI.PercentTrialsEasy;
        PercentTrialsMediumEasy = S.GUI.PercentTrialsMediumEasy;
        PercentTrialsMediumHard = S.GUI.PercentTrialsMediumHard;
        PercentTrialsHard = S.GUI.PercentTrialsHard;

        sum = PercentTrialsEasy + PercentTrialsMediumEasy + PercentTrialsMediumHard + PercentTrialsHard;

        FractionEasy = PercentTrialsEasy / sum;
        FractionMediumEasy = PercentTrialsMediumEasy / sum;
        FractionMediumHard = PercentTrialsMediumHard / sum;
        FractionHard = PercentTrialsHard / sum;
        
        % assigned probability weights for sampling from distribution
        %DifficultyProbabilities = [FractionEasy, FractionMedium, FractionHard];
        DifficultyProbabilities = [FractionEasy, FractionMediumEasy, FractionMediumHard, FractionHard];
        

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

    %% Draw trial-specific and difficulty-defined TimeOutPunish from exponential distribution

    [TimeOutPunish] = m_TrialConfig.GetTimeOutPunish(S);

    ExperimenterTrialInfo.TimeOutPunish = TimeOutPunish;


    %% Draw trial-specific ITI from exponential distribution

    [ITI] = m_TrialConfig.GetITI(S);

    ExperimenterTrialInfo.ITI = ITI;


    %% construct preperturb vis stim videos and audio stim base for grating and gray if duration parameters changed
    
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
    % if stim or isi dur changed, initial video pattern has to be
    % reconstructed from parts above
    if (S.GUI.GratingDur_s ~= LastGratingDuration) || (S.GUI.ISIOrig_s ~= LastGrayFixedDuration) || (S.GUI.NumISIOrigRep ~= LastNumISIOrigRep)
        VideoPrePerturbPattern = [GratingVideo GrayVideo]; % base video pattern for initial segment repetitions of grating->gray
        VideoData = [repmat(VideoPrePerturbPattern, 1, S.GUI.NumISIOrigRep) GratingVideo]; % construct initial video segment, add one more grating video after initial repetitions
        LastGratingDuration = S.GUI.GratingDur_s; % Remember value of stim dur so that we only regenerate the grating video if parameter has changed
        LastGrayFixedDuration = S.GUI.ISIOrig_s; % Remember value of stim dur so that we only regenerate the grating video if parameter has changed
        LastNumISIOrigRep = S.GUI.NumISIOrigRep;  % Remember value of initial segment repetitions so that we only regenerate the initial segment if parameter changed
    end
       

    %% set vis stim perturbation ISI duration according to trial-specific difficulty level
    % draw perturbation interval from uniform distribution in range according to difficulty

    EasyMinPercent = 3/4;
    EasyMaxPercent = 1;
    MediumEasyMinPercent = 1/2;
    MediumEasyMaxPercent = 3/4;
    MediumHardMinPercent = 1/4;
    MediumHardMaxPercent = 1/2;
    HardMinPercent = 0;
    HardMaxPercent = 1/4;
   
    % calculate category boundary to use as base timing reference for
    % perturbation length
    CatBound = S.GUI.ISIOrig_s;   % updated category boundary to be ISI_perturb * 1.5

    ExperimenterTrialInfo.CategoryBoundary = CatBound;   % capture variable states as field/value struct for experimenter info

    % full range of perturbation is calculated as the category boundary time
    % minus the selected minimum time separation from the category boundary and the grating

    PerturbDurFullRange = CatBound * 1000 - S.GUI.MinISIPerturb_ms; 
                                                                                                     
    switch TrialDifficulty % Determine trial-specific visual stimulus
        case 1
            PerturbDurMin = EasyMinPercent*PerturbDurFullRange;
            PerturbDurMax = EasyMaxPercent*PerturbDurFullRange;
        case 2
            PerturbDurMin = MediumEasyMinPercent*PerturbDurFullRange;
            PerturbDurMax = MediumEasyMaxPercent*PerturbDurFullRange;

        case 3
            PerturbDurMin = MediumHardMinPercent*PerturbDurFullRange;
            PerturbDurMax = MediumHardMaxPercent*PerturbDurFullRange;            
        
        case 4
            PerturbDurMin = HardMinPercent*PerturbDurFullRange;
            PerturbDurMax = HardMaxPercent*PerturbDurFullRange;
    end

    % after applying difficulty to perturbation range, shift the min and
    % max away to the selected minimum from category boundary
    % PerturbDurMin = PerturbDurMin + S.GUI.PerturbMinFromCB_ms;
    % PerturbDurMax = PerturbDurMax + S.GUI.PerturbMinFromCB_ms;

    debugDiffRanges = 1;
    if debugDiffRanges
        EasyMin = (EasyMinPercent*PerturbDurFullRange)/1000;
        EasyMax = (EasyMaxPercent*PerturbDurFullRange)/1000;
        MediumEasyMin = (MediumEasyMinPercent*PerturbDurFullRange)/1000;
        MediumEasyMax = (MediumEasyMaxPercent*PerturbDurFullRange)/1000;
        MediumHardMin = (MediumHardMinPercent*PerturbDurFullRange)/1000;
        MediumHardMax = (MediumHardMaxPercent*PerturbDurFullRange)/1000;        
        HardMin = (HardMinPercent*PerturbDurFullRange)/1000;
        HardMax = (HardMaxPercent*PerturbDurFullRange)/1000;
    end
    
    % EasyMax activation.
    % default: naive/mid1 is activated. mid2/well is deactivated.
    [RandomPerturbationDur, EasyMaxInfo] = m_TrialConfig.GetPerturbationDur( ...
        S, EasyMax, PerturbDurMin, PerturbDurMax);
    ExperimenterTrialInfo.EasyMax = EasyMaxInfo;

    % record trial unadjusted ISI random dur 
    BpodSystem.Data.TrialVars.Trial{currentTrial}.RandomPerturbationDur = RandomPerturbationDur;

    ExperimenterTrialInfo.DistanceFromCategoryBoundary = RandomPerturbationDur;   % capture variable states as field/value struct for experimenter info

    % set short and long ISI according to contingency defined by gui params
    % maybe collapse this into smaller if conditions
    switch TrialTypes(currentTrial) % Determine trial-specific visual stimulus duration
        case 1 % trial is left with short ISI                    
            GrayPerturbDuration = CatBound - RandomPerturbationDur; % for left reward, ISI is subtractd from the random duration            
        case 2 % trial is right with long ISI                    
            GrayPerturbDuration = CatBound + RandomPerturbationDur; % for right reward, ISI is added to the random duration
    end
    
    m_Plotter.UpdateSideOutcomePlot(BpodSystem, TrialTypes, 0); % this will update the SideOutcomePlot to reflect the current trial type after accounting for any change due to anti-bias control 
  
    debugGrayPerturbDuration = GrayPerturbDuration;


    %% Construct trial-specific portion of video and add it to base video

    % construct video for this trial
    
    % find number of frames for variable ISI    
    GrayPerturbFrames = convergent(FramesPerSecond * GrayPerturbDuration); % rounds ties to the nearest even integer

    if (mod(GrayPerturbFrames, 2) ~= 0)
        GrayPerturbFrames = GrayPerturbFrames + 1; % round up to nearest even integer
    end
    GrayPerturbVideo = [repmat(GrayPattern, 1, GrayPerturbFrames/2)];

    GrayPerturbDuration = GrayPerturbFrames / FramesPerSecond; % convert even rounded number of frames back into duration to calculate video duration

    ExperimenterTrialInfo.ISIPerturbDuration = GrayPerturbDuration;   % capture variable states as field/value struct for experimenter info

    VideoPerturbBasePattern = [GrayPerturbVideo GratingVideo]; % perturbation video pattern for second segment repetitions of random ISI gray->grating

    % find the nearest whole number duration of perturbation gray->grating that fits
    % within the scaled duration of the pre-preturbation segment
    PerturbBasePatternDuration = GrayPerturbDuration + GratingDuration;
    ScaledPrePerturbDuration = ((length(VideoData)-GratingFrames) / FramesPerSecond) * S.GUI.PostPerturbDurMultiplier;
    NumPerturbVisRep = floor(ScaledPrePerturbDuration/PerturbBasePatternDuration);
   
    %NumPerturbVisRep_frame_remainder = floor((ScaledPrePerturbDuration/PerturbBasePatternDuration - fix(ScaledPrePerturbDuration/PerturbBasePatternDuration))*FramesPerSecond);
    NumPerturbVisRep_frame_remainder = floor((ScaledPrePerturbDuration/PerturbBasePatternDuration - fix(ScaledPrePerturbDuration/PerturbBasePatternDuration))* length(VideoPerturbBasePattern));
    if NumPerturbVisRep_frame_remainder > 0
        VideoPerturbBasePattern_remaining_frames = VideoPerturbBasePattern(1:NumPerturbVisRep_frame_remainder);
    else
        VideoPerturbBasePattern_remaining_frames = [];
    end

    % use extra perturbation video for naive to extend vis stim to ITI
    switch S.GUI.TrainingLevel
        case 1 % Habituation       
            NumExtraPerturbVisRep = 90;   
        case 2 % Naive       
            NumExtraPerturbVisRep = 90;        
        case 3 % Mid Trained 1
            NumExtraPerturbVisRep = 90;
        case 4 % Mid Trained 2
            NumExtraPerturbVisRep = 90;
        case 5 % Trained
            NumExtraPerturbVisRep = 0;
    end

    NumInitialBaseFrames = GratingFrames + GrayFixedFrames;
    NumPerturbBaseFrames = GrayPerturbFrames + GratingFrames;

    NumPerturbReps = NumPerturbVisRep + NumExtraPerturbVisRep;

    TotalFramesNeeded = (S.GUI.NumISIOrigRep * NumInitialBaseFrames) + ...
        (S.GUI.NumISIOrigRep * NumInitialBaseFrames) * S.GUI.PostPerturbDurMultiplier + ...
        GratingFrames; % number of frames needed is     
    GrayFillerFramesNeeded = TotalFramesNeeded - (length(VideoData) + (length(VideoPerturbBasePattern)*NumPerturbReps));
    
    GrayFillerFramesNeeded = round(GrayFillerFramesNeeded); % get integer number of frames
    if (mod(GrayFillerFramesNeeded, 2) ~= 0)
        GrayFillerFramesNeeded = GrayFillerFramesNeeded + 1; % round up to nearest even integer
    end

    VideoGrayFiller = [repmat(GrayPattern, 1, GrayFillerFramesNeeded/2)];

    switch S.GUI.TrainingLevel
        case 1   
            FullVideo = [VideoData repmat(VideoPerturbBasePattern, 1, NumPerturbReps) VideoPerturbBasePattern_remaining_frames GrayBlank];
        case 2
            FullVideo = [VideoData repmat(VideoPerturbBasePattern, 1, NumPerturbReps) VideoPerturbBasePattern_remaining_frames GrayBlank];           
        case 3
            FullVideo = [VideoData repmat(VideoPerturbBasePattern, 1, NumPerturbReps) VideoPerturbBasePattern_remaining_frames GrayBlank];
        case 4
            FullVideo = [VideoData repmat(VideoPerturbBasePattern, 1, NumPerturbReps) VideoPerturbBasePattern_remaining_frames GrayBlank];
        case 5
            FullVideo = [VideoData repmat(VideoPerturbBasePattern, 1, NumPerturbReps) VideoGrayFiller];
    end

    if GrayFillerFramesNeeded < 0
        GrayFillerFramesNeeded = 0;
    end

    if (S.GUI.TrainingLevel == 1 || S.GUI.TrainingLevel == 2 || S.GUI.TrainingLevel == 3 || S.GUI.TrainingLevel == 4)
        FullVideoFrames = S.GUI.NumISIOrigRep * NumInitialBaseFrames + ...
            GratingFrames + ...
            NumPerturbVisRep * NumPerturbBaseFrames + ...
            NumExtraPerturbVisRep * NumPerturbBaseFrames + ...
            NumPerturbVisRep_frame_remainder; % + GratingFrame for the grating between base and variable segments of video and + 1 for final frame
    else
        FullVideoFrames = S.GUI.NumISIOrigRep * NumInitialBaseFrames + ...
            GratingFrames + ...
            NumPerturbVisRep * NumPerturbBaseFrames + ...
            NumExtraPerturbVisRep * NumPerturbBaseFrames + ...
            GrayFillerFramesNeeded; % + GratingFrame for the grating between base and variable segments of video and + 1 for final frame
    end

    FullVideoFrames = length(FullVideo);

    % this can be updated to merge grayfiller and remainder, they're the
    % same thing (are strictly unique though)
    % naive and mid 1 and mid 2: remainder perturb frames are used for fraction
    % of repitition remaining till go cue
    % well: gray filler frames are used till go cue to
    % prevent partial grating during go cue
    if (S.GUI.TrainingLevel == 1 || S.GUI.TrainingLevel == 2 || S.GUI.TrainingLevel == 3 || S.GUI.TrainingLevel == 4)
        VideoStartToGoCueFrames = S.GUI.NumISIOrigRep * NumInitialBaseFrames + ...
            NumPerturbVisRep * NumPerturbBaseFrames + ...
            GratingFrames + ...
            NumPerturbVisRep_frame_remainder;
            % 2*NumISIOrigRep for the repeated grating-gray pattern until go cue, + GratingFrame for the grating between base and variable segments of video and + 1 for final frame    
    else
        VideoStartToGoCueFrames = S.GUI.NumISIOrigRep * NumInitialBaseFrames + ...
            NumPerturbVisRep * NumPerturbBaseFrames + ...
            GratingFrames + ...
            GrayFillerFramesNeeded; % 2*NumISIOrigRep for the repeated grating-gray pattern until go cue, + GratingFrame for the grating between base and variable segments of video and + 1 for final frame            
    end

    VisStimDuration = VideoStartToGoCueFrames / FramesPerSecond; % calculate duration of vis stim based on when go cue state should begin after
                                                                    % start of vis stim state

    % load constructed video into the video object
    BpodSystem.PluginObjects.V.Videos{5} = struct;
    BpodSystem.PluginObjects.V.Videos{5}.nFrames = FullVideoFrames; 
    BpodSystem.PluginObjects.V.Videos{5}.Data = FullVideo;

    
    %% Generate audio stim based on vis stim for this trial

    % build audio stim to match grating/gray pattern
    PrePerturbNoSoundOffset = 1102;
    PrePerturbNoSoundOffset = 0;

    PostPerturbNoSoundOffset = 2205;
    PostPerturbNoSoundOffset = 0;

    GratingNumSamples = GratingDur * SF;     
    GrayNumSamples = GrayDur * SF;  % get duration of gray in number of audio samples for period between audio stim

    NoSoundPrePerturb = zeros(1, GrayNumSamples+PrePerturbNoSoundOffset);

    AudioPrePerturbPattern = [AudioStimSound NoSoundPrePerturb];
    PrePerturbAudioData = [repmat(AudioPrePerturbPattern, 1, S.GUI.NumISIOrigRep) AudioStimSound]; % construct preperturb audio
    
    GrayPerturbDur = length(GrayPerturbVideo) * (1/FramesPerSecond);  % get duration of gray perturb
    GrayPerturbNumSamples = floor(GrayPerturbDur * SF);    % get duration of gray perturb in number of audio samples for period of grating
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
            FullAudioStimData = FullAudioStimData + ShiftedGoCue;
        case 2
            VideoStartToGoCueDur_s = VideoStartToGoCueFrames * (1/FramesPerSecond);
            VideoStartToGoCueDur_NumAudioSamples = SF * VideoStartToGoCueDur_s;
            ShiftedGoCue = [zeros(1, VideoStartToGoCueDur_NumAudioSamples) GoCueSound zeros(1, length(FullAudioStimData) - VideoStartToGoCueDur_NumAudioSamples - length(GoCueSound))];
            FullAudioStimData = FullAudioStimData + ShiftedGoCue;
        case 3
            VideoStartToGoCueDur_s = VideoStartToGoCueFrames * (1/FramesPerSecond);
            VideoStartToGoCueDur_NumAudioSamples = SF * VideoStartToGoCueDur_s;
            ShiftedGoCue = [zeros(1, VideoStartToGoCueDur_NumAudioSamples) GoCueSound zeros(1, length(FullAudioStimData) - VideoStartToGoCueDur_NumAudioSamples - length(GoCueSound))];
            FullAudioStimData = FullAudioStimData + ShiftedGoCue;
        case 4
            VideoStartToGoCueDur_s = VideoStartToGoCueFrames * (1/FramesPerSecond);
            VideoStartToGoCueDur_NumAudioSamples = SF * VideoStartToGoCueDur_s;
            ShiftedGoCue = [zeros(1, VideoStartToGoCueDur_NumAudioSamples) GoCueSound zeros(1, length(FullAudioStimData) - VideoStartToGoCueDur_NumAudioSamples - length(GoCueSound))];
            FullAudioStimData = FullAudioStimData + ShiftedGoCue;
        case 5
    end

    H.load(5, FullAudioStimData);

    % Query duration of one monitor refresh interval:
    ifi=Screen('GetFlipInterval', BpodSystem.PluginObjects.V.Window);
    

    %% update trial-specific Audio

    %H.DigitalAttenuation_dB = S.GUI.DigitalAttenuation_dB; % update sound level to param GUI
    if S.GUI.IncorrectSound
        OutputActionArgIncorrect = {'HiFi1', ['P' 2]};
    else
        OutputActionArgIncorrect = {};
    end

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

    if (S.GUI.PunishSoundDuration_s ~= LastPunishSoundDuration) || ...
        (S.GUI.IncorrectSoundVolume_percent ~= LastIncorrectSoundVolume)
        IncorrectSound = GenerateWhiteNoise(SF, S.GUI.PunishSoundDuration_s, 1, 2)*S.GUI.IncorrectSoundVolume_percent; % white noise punish sound
        IncorrectSound = ApplySoundEnvelope(IncorrectSound, Envelope);
        H.load(3, IncorrectSound);
        LastPunishSoundDuration = S.GUI.PunishSoundDuration_s;
        LastIncorrectSoundVolume = S.GUI.IncorrectSoundVolume_percent;
    end
     

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

    OutputActionsPreGoCueDelay = {}; % 
    OutputActionsEarlyChoice = {'SoftCode', 255, 'HiFi1', 'X'}; % stop audio stim, stop vis stim
    OutputActionsPunishSetup = {'SoftCode', 255, 'HiFi1', 'X'};
    visStim = {'SoftCode', 5};
    audStim = {'HiFi1', ['P', 4], 'BNCState', 1};


    %% training level specific state matrix values

    switch S.GUI.TrainingLevel

        case 1 % Habituation        
            InitCue_Tup_NextState = 'InitReward'; % change to fn_lic_poiss_3              
            VisualStimulusStateChangeConditions = {'Tup', 'CenterReward'};
            CenterReward_OutputActions = {'Valve2', 1};
            CenterReward_NextSate = 'HabituationExtendWindow';
            GoCue_Tup_NextState = 'RewardNaive';  % naive
            WindowChoice_StateChangeConditions = {};
            OutputActionsWindowChoice = {};
            Reward_Tup_NextState = 'ITI';
            PunishSetup_Tup_NextState = 'PunishNaive'; % Naive
            stimDuration = VisStimDuration;
            ExperimenterTrialInfo.TrainingLevel = 'Habituation';

        case 2 % naive        
            InitCue_Tup_NextState = 'InitWindow'; % change to fn_lic_poiss_3              
            VisualStimulusStateChangeConditions = {'Tup', 'CenterReward'};
            CenterReward_OutputActions = {'Valve2', 1};
            CenterReward_NextSate = 'PreGoCueDelay';
            GoCue_Tup_NextState = 'RewardNaive';  % naive
            WindowChoice_StateChangeConditions = {};
            OutputActionsWindowChoice = {};
            Reward_Tup_NextState = 'ITI';
            PunishSetup_Tup_NextState = 'PunishNaive'; % Naive
            stimDuration = VisStimDuration;
            ExperimenterTrialInfo.TrainingLevel = 'Naive';

        case 3 % Mid 1 Trained
            InitCue_Tup_NextState = 'InitWindow';
            VisualStimulusStateChangeConditions = {'Tup', 'CenterReward'};
            CenterReward_OutputActions = {'Valve2', 1}; % moved video stop code earlier to center reward
            CenterReward_NextSate = 'PreGoCueDelay';
            GoCue_Tup_NextState = 'WindowChoice';  % trained
            WindowChoice_StateChangeConditions = {CorrectLick, 'RewardDelay', 'Condition1', 'RewardDelay', IncorrectLick, 'PunishSetup', 'Condition2', 'PunishSetup', 'Tup', 'DidNotChoose'};
            OutputActionsWindowChoice = {};
            Reward_Tup_NextState = 'ExtraStimDurPostRew_Naive';
            PunishSetup_Tup_NextState = 'Punish'; % trained
            stimDuration = VisStimDuration;
            ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 1';

        case 4 % Mid 2 Trained
            InitCue_Tup_NextState = 'InitWindow';
            VisualStimulusStateChangeConditions = {'Tup', 'VisualStim_AllowedSideLick', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
            CenterReward_OutputActions = {'Valve2', 1}; % moved video stop code earlier to center reward
            CenterReward_NextSate = 'PreGoCueDelay';
            GoCue_Tup_NextState = 'WindowChoice';  % trained
            WindowChoice_StateChangeConditions = {CorrectLick, 'RewardDelay', 'Condition1', 'RewardDelay', IncorrectLick, 'PunishSetup', 'Condition2', 'PunishSetup', 'Tup', 'DidNotChoose'};
            OutputActionsWindowChoice = {};
            Reward_Tup_NextState = 'ExtraStimDurPostRew_Naive';
            PunishSetup_Tup_NextState = 'Punish'; % trained
            stimDuration = wait_dur;
            ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 2';

        case 5 % well trained
            InitCue_Tup_NextState = 'InitWindow';
            VisualStimulusStateChangeConditions = {'Tup', 'VisualStim_AllowedSideLick', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'};
            CenterReward_OutputActions = {'Valve2', 1, 'SoftCode', 255}; % moved video stop code earlier to center reward
            CenterReward_NextSate = 'PreGoCueDelay';
            GoCue_Tup_NextState = 'WindowChoice';  % trained
            OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};
            WindowChoice_StateChangeConditions = {CorrectLick, 'CorrectLickInterval', 'Condition1', 'CorrectLickInterval', IncorrectLick, 'IncorrectLickInterval', 'Condition2', 'IncorrectLickInterval', 'Tup', 'DidNotChoose'};
            OutputActionsWindowChoice = {'HiFi1', 'X'};
            Reward_Tup_NextState = 'ITI';
            PunishSetup_Tup_NextState = 'Punish'; % trained
            stimDuration = wait_dur;
            ExperimenterTrialInfo.TrainingLevel = 'Well Trained';
    end


    %%
    ExperimenterTrialInfo.VisStimDuration = VisStimDuration;


    %% After warmup trials, reset wait_dur: with every non early-choice trial, increase it by wait_dur_step

    [wait_dur] = m_TrialConfig.GetWaitDur(BpodSystem, S, wait_dur, currentTrial, LastNumEasyWarmupTrials, VisStimDuration);
    
    ExperimenterTrialInfo.WaitDuration = wait_dur;


    %% add console print for experimenter trial information

    strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
    disp(strExperimenterTrialInfo);


    %% construct state matrix

    sma = NewStateMatrix(); % Assemble state matrix
    % needs more conditions here when adding difficulty levels
    sma = SetCondition(sma, 1, CorrectPort, 1); % Condition 1: Correct Port is high (licking)
    sma = SetCondition(sma, 2, IncorrectPort, 1); % Condition 2: Incorrect Port is high (licking)

    % testing behavior port4in port4out for use as sync signal from camera
    sma = SetCondition(sma, 3, 'Port4', 1); % Condition 3: Sync signal high
    sma = SetCondition(sma, 4, 'Port4', 0); % Condition 4: Sync signal low
    sma = SetCondition(sma, 5, 'Port2', 1); % Condition 5: center port is high (licking)

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
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...         
        'OutputActions', {});     

    % delay after init before visual stimulus
    % placeholder - not in  use
    sma = AddState(sma, 'Name', 'PreVisStimDelay', ...
        'Timer', S.GUI.PreVisStimDelay_s,...
        'StateChangeConditions', {'Tup', 'VisStimTrigger'},...
        'OutputActions', {});    

    sma = AddState(sma, 'Name', 'VisStimTrigger', ...
        'Timer', 0,...
        'StateChangeConditions', {'BNC1High', 'AudStimTrigger'},...
        'OutputActions', visStim);

    % VisualStimulus : naive and mid-trained
    sma = AddState(sma, 'Name', 'AudStimTrigger', ...
        'Timer', stimDuration,...
        'StateChangeConditions', VisualStimulusStateChangeConditions,...
        'OutputActions', audStim);

    % VisualStimulus; they can side lick (though they wont get reward bc
    % it's before go cue)
    sma = AddState(sma, 'Name', 'VisualStim_AllowedSideLick', ...
        'Timer', VisStimDuration - wait_dur,...
        'StateChangeConditions', {'Tup', 'CenterReward'},...
        'OutputActions', {});

    % Well trained: after the stimulus ends, mouse center licks, then center reward happens.  
    sma = AddState(sma, 'Name', 'CenterReward', ...
        'Timer', CenterValveTime,...
        'StateChangeConditions', {'Tup', CenterReward_NextSate},...
        'OutputActions', CenterReward_OutputActions);    

    % placeholder - not in use
    sma = AddState(sma, 'Name', 'PreGoCueDelay', ...
        'Timer', S.GUI.PreGoCueDelay_s,...
        'StateChangeConditions', {'Tup', 'GoCue', 'Port1In', 'EarlyChoice', 'Port3In', 'EarlyChoice'},...         
        'OutputActions', OutputActionsPreGoCueDelay);

    % also stop video here, there can be small delay even with zero timer
    % when waiting to ITI
    sma = AddState(sma, 'Name', 'EarlyChoice', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', OutputActionsEarlyChoice);

    % well trained - mouse didn't lick center after vis stim during S.GUI.CenterLickWindow_Trained_s
    sma = AddState(sma, 'Name', 'EarlyChoiceDurCenterLick', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', {});

    % play go cue sound
    sma = AddState(sma, 'Name', 'GoCue', ...
        'Timer', S.GUI.GoCueDuration_s,...
        'StateChangeConditions', {'Tup', GoCue_Tup_NextState},...         
        'OutputActions', OutputActionArgGoCue);      

    sma = AddState(sma, 'Name', 'RewardNaive', ...
        'Timer', ValveTime,...
        'StateChangeConditions', {'Tup', 'WindowRewardGrab_Naive'},...
        'OutputActions', {Valve, 1});    
    
    % keep valve open here? no
    sma = AddState(sma, 'Name', 'WindowRewardGrab_Naive', ...
        'Timer', S.GUI.WindowRewardGrabDuration_Naive_s,...
        'StateChangeConditions', {'Tup', 'DidNotChoose', CorrectLick, 'ExtraStimDurPostRew_Naive', 'Condition1', 'ExtraStimDurPostRew_Naive', IncorrectLick, 'PunishSetup', 'Condition2', 'PunishSetup'},...
        'OutputActions', {});
    
    % additional time to associate stimulus with reward
    sma = AddState(sma, 'Name', 'ExtraStimDurPostRew_Naive', ...
        'Timer', S.GUI.ExtraStimDurPostRew_Naive_s,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});
    
    % naive - don't require confirm/choose again loop, log DidNotChoose,
    %         reward from correct spout anyway
    % trained - 
    % well trained - ?
    sma = AddState(sma, 'Name', 'WindowChoice', ...
        'Timer', S.GUI.ChoiceWindow_s,...
        'StateChangeConditions', WindowChoice_StateChangeConditions,...
        'OutputActions', OutputActionsWindowChoice);        
    
    % trained - did not lick center after vis stim for center reward
    sma = AddState(sma, 'Name', 'DidNotLickCenter', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', {});

    % naive - log DidNotChoose, skip confirm/choose again loop, go to
    %         reward correct spout
    % trained - log DidNotChoose, go to ITI
    % well trained - ?
    sma = AddState(sma, 'Name', 'DidNotChoose', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', {});   

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

    sma = AddState(sma, 'Name', 'DidNotConfirm', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
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

    sma = AddState(sma, 'Name', 'PunishNaive', ...
        'Timer', S.GUI.PunishSoundDuration_s,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', OutputActionArgIncorrect);

    sma = AddState(sma, 'Name', 'Punish', ...
        'Timer', S.GUI.PunishSoundDuration_s,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', OutputActionArgIncorrect);

    sma = AddState(sma, 'Name', 'TimeOutPunish', ...
        'Timer', TimeOutPunish,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'HabituationExtendWindow', ...
        'Timer', 6,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});

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
        m_Plotter.UpdateSideOutcomePlot(BpodSystem, TrialTypes, 1);
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

    %% Take care of WarmupTrialsCounter

    if WarmupTrialsCounter > 0
	    WarmupTrialsCounter = WarmupTrialsCounter - 1;
    end


end

BpodSystem.PluginObjects.V = [];
BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session


% generate full envelope for sound given the sound and front part of
% envelope, return enveloped sound
function [SoundWithEnvelope] = ApplySoundEnvelope(Sound, Envelope)
    BackOfTheEnvelope = fliplr(Envelope);   % flipe front envelope to get back envelope
    IdxsBetweenTheEnvelope = length(Sound) - 2 * length(Envelope); % indices between front and back of envelope
    FullEnvelope = [Envelope ones(1, IdxsBetweenTheEnvelope) BackOfTheEnvelope];  % full envelope
    SoundWithEnvelope = Sound .* FullEnvelope;    % apply envelope element-wise


