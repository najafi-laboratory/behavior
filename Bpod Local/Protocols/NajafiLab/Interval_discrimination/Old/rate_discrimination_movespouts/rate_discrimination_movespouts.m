function rate_discrimination_movespouts

global BpodSystem
global M

EnableMovingSpouts = 1;
EnablePassive      = 0;
MonitorID          = 2;


%% Import scripts

m_Plotter      = Plotter;
m_InitGUI      = InitGUI;
m_TrialConfig  = TrialConfig;
m_AVstimConfig = AVstimConfig;


%% Turn off Bpod LEDs

% This code will disable the state machine status LED
BpodSystem.setStatusLED(0);

%% Assert HiFi module is present + USB-paired (via USB button on console GUI)

BpodSystem.assertModule('HiFi', 1); % The second argument (1) indicates that the HiFi module must be paired with its USB serial port
% Create an instance of the HiFi module
H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1); % The argument is the name of the HiFi module's USB serial port (e.g. COM3)


%% Define parameters
global S
[S] = m_InitGUI.SetParams(BpodSystem);
[S] = m_InitGUI.UpdateMovingSpouts(S, EnableMovingSpouts);
[S] = m_InitGUI.UpdatePassive(S, EnablePassive);


%% Connect Maestro
if (EnableMovingSpouts == 1)
    % Where COM3 is the Maestro USB serial command port
    M = PololuMaestro('COM13');
    % move servos to out position for start
    M.setMotor(0, m_TrialConfig.ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
    M.setMotor(1, m_TrialConfig.ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));
end


%% Define trials

% set max number of trials
MaxTrials = 1000;

% initialize anti-bias variables
AntiBiasVar.IncorrectFlag       = 0;
AntiBiasVar.IncorrectType       = 1;
AntiBiasVar.CompletedHist.left  = [];
AntiBiasVar.CompletedHist.right = [];
AntiBiasVar.ValveFlag           = 'NoBias';

% draw perturbation interval from uniform distribution in range according to difficulty
PerturbInterval.EasyMinPercent       = 3/4;
PerturbInterval.EasyMaxPercent       = 1;
PerturbInterval.MediumEasyMinPercent = 1/2;
PerturbInterval.MediumEasyMaxPercent = 3/4;
PerturbInterval.MediumHardMinPercent = 1/4;
PerturbInterval.MediumHardMaxPercent = 1/2;
PerturbInterval.HardMinPercent       = 0;
PerturbInterval.HardMaxPercent       = 1/4;

% get uniform distribution of 2 trial types
[TrialTypes] = m_TrialConfig.GenTrials(MaxTrials);
[TrialTypes] = m_TrialConfig.AdjustWarmupTrials(S, TrialTypes);

% override trials when passive is activated
if (EnablePassive == 1)
    [TrialTypes] = m_TrialConfig.GenPassiveTrials(S, MaxTrials);
end

BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.


%% Initialize plots

% Side Outcome Plot
BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes);
BpodParameterGUI('init', S); % Initialize parameter GUI plugin

useStateTiming = true;  % Initialize state timing plot
if ~verLessThan('matlab','9.5') % StateTiming plot requires MATLAB r2018b or newer
    useStateTiming = true;
    StateTiming();
end


%% Define stimuli and send to analog module

SF = 44100; % Use lower sampling rate (samples/sec) to allow for longer duration audio file (max length limited by HiFi)
H.SamplingRate = SF;
Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset

H.DigitalAttenuation_dB = -35; % Set a comfortable listening level for most headphones (useful during protocol dev).

m_AVstimConfig.ConfigHifi(H, S, SF, Envelope);


%% Setup video

if isfield(BpodSystem.PluginObjects, 'V') % Clear previous instances of the video server
    BpodSystem.PluginObjects.V = [];
end
BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(MonitorID, 0, [0 0], [180 180], 0); % Assumes Sync patch = 180x180 pixels
BpodSystem.PluginObjects.V.SyncPatchIntensity = 255;
BpodSystem.PluginObjects.V.loadText(1, 'Loading...', '', 80);
BpodSystem.PluginObjects.V.play(1);
Xsize = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
Ysize = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
FPS   = BpodSystem.PluginObjects.V.DetectedFrameRate;

% generate frame images
ImgParams.spatialFreq = .005;
ImgParams.orientation = 0;
ImgParams.contrast    = 1;
ImgParams.phase       = 0.5;
[VideoGrating, VideoGrayFixed] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
BpodSystem.PluginObjects.V.loadVideo(1, VideoGrating);
BpodSystem.PluginObjects.V.loadVideo(2, VideoGrayFixed);

% compost grating video
VisStim.Img.GratingFrame_SyncW = BpodSystem.PluginObjects.V.Videos{1}.Data(1);
VisStim.Img.GratingBlank       = BpodSystem.PluginObjects.V.Videos{1}.Data(3);
VisStim.Img.GrayFrame_SyncW    = BpodSystem.PluginObjects.V.Videos{2}.Data(1);
VisStim.Img.GrayFrame_SyncBlk  = BpodSystem.PluginObjects.V.Videos{2}.Data(2);
VisStim.Img.GrayBlank          = BpodSystem.PluginObjects.V.Videos{2}.Data(3);
VisStim.Grating.Frames = m_AVstimConfig.GetFrames(FPS, S.GUI.GratingDur_s);
VisStim.GrayPre.Frames = m_AVstimConfig.GetFrames(FPS, S.GUI.ISIOrig_s);
VisStim.Grating.Video  = m_AVstimConfig.GetUnitVideo(VisStim.Img.GratingFrame_SyncW, VisStim.Grating.Frames);
VisStim.GrayPre.Video  = m_AVstimConfig.GetUnitVideo(VisStim.Img.GrayFrame_SyncBlk, VisStim.GrayPre.Frames);

BpodSystem.PluginObjects.V.Videos{3}         = struct;
BpodSystem.PluginObjects.V.Videos{4}         = struct;
BpodSystem.PluginObjects.V.Videos{6}         = struct;
BpodSystem.PluginObjects.V.Videos{3}.nFrames = VisStim.Grating.Frames + 1;
BpodSystem.PluginObjects.V.Videos{4}.nFrames = VisStim.GrayPre.Frames + 1;
BpodSystem.PluginObjects.V.Videos{6}.nFrames = 121;
BpodSystem.PluginObjects.V.Videos{3}.Data    = [VisStim.Grating.Video VisStim.Img.GratingBlank];
BpodSystem.PluginObjects.V.Videos{4}.Data    = [VisStim.GrayPre.Video VisStim.Img.GrayBlank];
BpodSystem.PluginObjects.V.Videos{6}.Data    = [repmat(VisStim.Img.GrayFrame_SyncW, 1, 120) VisStim.Img.GrayFrame_SyncBlk];
BpodSystem.PluginObjects.V.TimerMode = 0;
pause(1.0);
BpodSystem.PluginObjects.V.play(0);
BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler';
BpodSystem.PluginObjects.V.play(6);
BpodSystem.PluginObjects.V.TimerMode = 2;
input('Set parameters and press enter to continue >', 's'); 
S = BpodParameterGUI('sync', S);


%% check difficulty options and ensure correct setting prior to beginning first trial

Warmup.Flag = true;
Warmup.Res = S.GUI.NumEasyWarmupTrials;

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
    ExperimenterTrialInfo.Bias           = AntiBiasVar.ValveFlag;
    ExperimenterTrialInfo.LeftValveTime  = LeftValveTime;
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
    
    [Warmup, TrialDifficulty] = m_TrialConfig.DiffHandler(S, Warmup);

    ExperimenterTrialInfo.Warmup = Warmup.Flag;
    ExperimenterTrialInfo.WarmupRes = Warmup.Res;
    switch TrialDifficulty
        case 1
            ExperimenterTrialInfo.Difficulty = 'Easy';
        case 2
            ExperimenterTrialInfo.Difficulty = 'Medium-Easy';
        case 3
            ExperimenterTrialInfo.Difficulty = 'Medium-Hard';
        case 4
            ExperimenterTrialInfo.Difficulty = 'Hard';
    end

    
    %% Draw trial-specific and difficulty-defined TimeOutPunish from exponential distribution

    [TimeOutPunish] = m_TrialConfig.GetTimeOutPunish(S);
    ExperimenterTrialInfo.TimeOutPunish = TimeOutPunish;


    %% Draw trial-specific ITI from exponential distribution

    [ITI] = m_TrialConfig.GetITI(S);
    ExperimenterTrialInfo.ITI = ITI;


    %% set vis stim perturbation ISI duration according to trial-specific difficulty level

    % perturbation sampling
    if (S.GUI.EnablePassive == 0)
        [RandomPerturbationDur, EasyMaxInfo] = m_TrialConfig.GetPerturbationDur( ...
            S, TrialDifficulty, PerturbInterval);
        [GrayPerturbISI] = m_TrialConfig.GetPerturbDur( ...
        S, TrialTypes, currentTrial, RandomPerturbationDur);
    else
        [RandomPerturbationDur, EasyMaxInfo] = m_TrialConfig.GetPerturbationDurPassive( ...
        S, currentTrial, TrialDifficulty, PerturbInterval);
        [GrayPerturbISI] = m_TrialConfig.GetPerturbDurPassive( ...
        S, TrialTypes, currentTrial, RandomPerturbationDur);
    end
    ExperimenterTrialInfo.CategoryBoundary = S.GUI.ISIOrig_s;
    ExperimenterTrialInfo.EasyMax = EasyMaxInfo;

    BpodSystem.Data.TrialVars.Trial{currentTrial}.RandomPerturbationDur = RandomPerturbationDur;

    m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 0);


    %% construct preperturb vis stim videos and audio stim base for grating and gray if duration parameters changed
    
    VisStim.Grating.Frames  = m_AVstimConfig.GetFrames(FPS, S.GUI.GratingDur_s);
    VisStim.GrayPre.Frames  = m_AVstimConfig.GetFrames(FPS, S.GUI.ISIOrig_s);
    VisStim.GrayPost.Frames = m_AVstimConfig.GetFrames(FPS, GrayPerturbISI);
    VisStim.Grating.Video   = m_AVstimConfig.GetUnitVideo(VisStim.Img.GratingFrame_SyncW, VisStim.Grating.Frames);
    VisStim.GrayPre.Video   = m_AVstimConfig.GetUnitVideo(VisStim.Img.GrayFrame_SyncBlk, VisStim.GrayPre.Frames);
    VisStim.GrayPost.Video  = m_AVstimConfig.GetUnitVideo(VisStim.Img.GrayFrame_SyncBlk, VisStim.GrayPost.Frames);
    VisStim.Grating.Dur     = length(VisStim.Grating.Video) * (1/FPS);
    VisStim.GrayPre.Dur     = length(VisStim.GrayPre.Video) * (1/FPS);
    VisStim.GrayPost.Dur    = VisStim.GrayPost.Frames / FPS;
    if (S.GUI.EnablePassive == 0)
        VisStim.Data.Pre    = m_AVstimConfig.GetVideoDataPre(S, VisStim.Grating.Video, VisStim.GrayPre.Video, 0);
    else
    end
    [PreFrames]    = m_AVstimConfig.GetPreFrames(S, VisStim);
    [PostFrames]   = m_AVstimConfig.GetPostFrames(S, VisStim);
    [NumExtraStim] = m_AVstimConfig.GetNumExtraStim(S);
    [TotalFrames]  = m_AVstimConfig.GetTotalFrames(S, VisStim);
    [AudioStimSound] = m_AVstimConfig.GenAudioStim(S, SF, Envelope);


    %% Construct trial-specific portion of video and add it to base video

    ExperimenterTrialInfo.ISIPerturbDuration = VisStim.GrayPost.Dur;

    PertVideo = [VisStim.GrayPost.Video VisStim.Grating.Video];

    PostPertDur = VisStim.GrayPost.Dur + VisStim.Grating.Dur;
    PrePertDur = ((length(VisStim.Data.Pre)-VisStim.Grating.Frames) / FPS) * S.GUI.PostPerturbDurMultiplier;
    NumPerturbVisRep = floor(PrePertDur/PostPertDur);
   
    NumPerturbReps = NumPerturbVisRep + NumExtraStim;

    [FillerFrames] = m_AVstimConfig.GetFillerFrames(TotalFrames, VisStim, PertVideo, NumPerturbReps);

    VisStim.Filler.Video = m_AVstimConfig.GetUnitVideo(VisStim.Img.GrayFrame_SyncBlk, FillerFrames);

    if (S.GUI.TrainingLevel == 2 || S.GUI.TrainingLevel == 3 || S.GUI.TrainingLevel == 4 || S.GUI.TrainingLevel == 5)
        VideoStartToGoCueFrames = PreFrames + ...
            NumPerturbVisRep * (VisStim.GrayPost.Frames + VisStim.Grating.Frames) + ...
            VisStim.Grating.Frames + ...
            NumPerRes;
    else
        VideoStartToGoCueFrames = PreFrames + ...
            NumPerturbVisRep * (VisStim.GrayPost.Frames + VisStim.Grating.Frames) + ...
            VisStim.Grating.Frames + ...
            FillerFrames;
    end

    VisStimDuration = VideoStartToGoCueFrames / FPS;

    VisStim.Data.Post = repmat(PertVideo, 1, NumPerturbReps);

    NumPerRes = floor((PrePertDur/PostPertDur - fix(PrePertDur/PostPertDur))* length(PertVideo));
    if NumPerRes > 0
        PertFramesRes = PertVideo(1:NumPerRes);
    else
        PertFramesRes = [];
    end

    [FullVideo] = m_AVstimConfig.GenFullVideo( ...
        S, VisStim, PertFramesRes, VisStim.Img.GrayBlank);

    % load constructed video into the video object
    BpodSystem.PluginObjects.V.Videos{5} = struct;
    BpodSystem.PluginObjects.V.Videos{5}.nFrames = length(FullVideo); 
    BpodSystem.PluginObjects.V.Videos{5}.Data = FullVideo;

    
    %% Generate audio stim based on vis stim for this trial

    m_AVstimConfig.ConfigFullAudioStim( ...
        H, S, VisStim.Grating.Dur, VisStim.GrayPre.Dur, SF, VideoStartToGoCueFrames, FPS, NumPerturbReps, ...
        AudioStimSound, VisStim.GrayPost.Video, Envelope);
    
    
    %% update trial-specific Audio

    if S.GUI.IncorrectSound
        OutputActionArgIncorrect = {'HiFi1', ['P' 2]};
    else
        OutputActionArgIncorrect = {};
    end
    

    %% update trial-specific state matrix fields

    % port 1: left (check on bPod console)
    % port 2: middle
    % port 3: right
    switch TrialTypes(currentTrial) % Determine trial-specific state matrix fields
        case 1 % short ISI % left side is rewarded
            CorrectLick = 'Port1In';
            CorrectPort = 'Port1';
            IncorrectLick = 'Port3In';
            IncorrectPort = 'Port3';
            ValveTime = LeftValveTime;
            Valve = 'Valve1';
            ExperimenterTrialInfo.CorrectChoice = 'Left';   % capture variable states as field/value struct for experimenter info
        case 2  % long ISI % right side is rewarded
            CorrectLick = 'Port3In';
            CorrectPort = 'Port3';
            IncorrectLick = 'Port1In';
            IncorrectPort = 'Port1';
            ValveTime = RightValveTime;
            Valve = 'Valve3';
            ExperimenterTrialInfo.CorrectChoice = 'Right';   % capture variable states as field/value struct for experimenter info
    end

    
    %% trial-specific output actions
    % maybe move this to state-matrix section for consistency
    OutputActionArgInitCue = {'HiFi1', ['P' 0]};
    [OutputActionArgGoCue] = m_AVstimConfig.GetOutputAct(S);
    
    if (EnableMovingSpouts == 1)
        OutputActionsPreGoCueDelay = {'SoftCode', 9};
    else
        OutputActionsPreGoCueDelay = {};
    end
    OutputActionsEarlyChoice = {'SoftCode', 255, 'HiFi1', 'X'}; % stop audio stim, stop vis stim
    OutputActionsPunishSetup = {'SoftCode', 255, 'HiFi1', 'X'};
    visStim = {'SoftCode', 5};
    audStim = {'HiFi1', ['P', 4], 'BNCState', 1, 'BNC1', 1};


    %%
    ExperimenterTrialInfo.VisStimDuration = VisStimDuration;


    %% After warmup trials, reset wait_dur: with every non early-choice trial, increase it by wait_dur_step

    [wait_dur] = m_TrialConfig.GetWaitDur(BpodSystem, S, wait_dur, currentTrial, VisStimDuration);
    
    ExperimenterTrialInfo.WaitDuration = wait_dur;


    %% add console print for experimenter trial information

    strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
    disp(strExperimenterTrialInfo);


    %% construct state matrix
    switch S.GUI.TrainingLevel
        case 1 % passive
            ExperimenterTrialInfo.TrainingLevel = 'Passive';
            StatePassive(S, visStim, audStim, VisStimDuration, ITI);
        case 2 % Habituation
            ExperimenterTrialInfo.TrainingLevel = 'Habituation';
            StateHabituation(S, OutputActionArgInitCue, ...
                CenterValveTime, visStim, VisStimDuration, audStim, OutputActionsPreGoCueDelay, ITI);
        case 3 % naive
            ExperimenterTrialInfo.TrainingLevel = 'Naive';
            StateNaive(S, CorrectPort, IncorrectPort, CorrectLick, IncorrectLick, ...
                OutputActionArgInitCue, CenterValveTime, ...
                visStim, VisStimDuration, audStim, OutputActionsPreGoCueDelay, ...
                OutputActionArgGoCue, ValveTime, Valve, ...
                OutputActionsPunishSetup, OutputActionArgIncorrect, TimeOutPunish, ITI);
        case 4 % Mid 1 Trained
            ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 1';
            StateMidTrain1(S, CorrectPort, IncorrectPort, CorrectLick, IncorrectLick, OutputActionArgInitCue, ...
                CenterValveTime, visStim, audStim, ...
                VisStimDuration, OutputActionsPreGoCueDelay, ...
                OutputActionArgGoCue, ValveTime, Valve, ...
                OutputActionsPunishSetup, OutputActionArgIncorrect, TimeOutPunish, ITI);
        case 5 % Mid 2 Trained
            ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 2';
            StateMidTrain2(S, CorrectPort, IncorrectPort, CorrectLick, IncorrectLick, OutputActionArgInitCue, ...
                CenterValveTime, visStim, audStim, ...
                VisStimDuration, wait_dur, OutputActionsPreGoCueDelay, ...
                OutputActionsEarlyChoice, OutputActionArgGoCue, ValveTime, Valve, ...
                OutputActionsPunishSetup, OutputActionArgIncorrect, TimeOutPunish, ITI);
        case 6 % well trained
            ExperimenterTrialInfo.TrainingLevel = 'Well Trained';
            StateWellTrained(S, CorrectPort, IncorrectPort, CorrectLick, IncorrectLick, OutputActionArgInitCue, ...
                CenterValveTime, visStim, audStim, ...
                VisStimDuration, wait_dur, OutputActionsPreGoCueDelay, ...
                OutputActionsEarlyChoice, OutputActionArgGoCue, ValveTime, Valve, ...
                OutputActionsPunishSetup, OutputActionArgIncorrect, TimeOutPunish, ITI);
    end

    RawEvents = RunStateMachine; % Run the trial and return events

    if ~isempty(fieldnames(RawEvents)) % If trial data was returned (i.e. if not final trial, interrupted by user)
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
        BpodSystem.Data.TrialSettings(currentTrial) = S; % Adds the settings used for the current trial to the Data struct (to be saved after the trial ends)
        BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial); % Adds the trial type of the current trial to data
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 1);
        if useStateTiming
            StateTiming();
        end
        SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file
    end
    HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
    if BpodSystem.Status.BeingUsed == 0 % If protocol was stopped, exit the loop
        clear global M; % disconnect maestro
        BpodSystem.PluginObjects.V = [];
        BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session?
        return
    end

end

clear global M; % disconnect maestro
BpodSystem.PluginObjects.V = [];
BpodSystem.setStatusLED(1); % enable Bpod status LEDs after session?


