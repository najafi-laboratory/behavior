function rate_discrimination_hilbert

global BpodSystem
global M


EnableMovingSpouts = 0;
EnablePassive      = 1;
PassiveSessMode    = 2; %1: omission; 2:ISI
MonitorID          = 2;


%% Import scripts

m_Plotter      = Plotter;
m_InitGUI      = InitGUI;
m_TrialConfig  = TrialConfig;
m_AVstimConfig = AVstimConfig;
m_PostProcess  = PostProcess;


%% Turn off Bpod LEDs

BpodSystem.setStatusLED(0);


%% Assert HiFi module is present + USB-paired (via USB button on console GUI)

BpodSystem.assertModule('HiFi', 1);
H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);


%% Define parameters

global S

[S] = m_InitGUI.SetParams(BpodSystem);
[S] = m_InitGUI.UpdateMovingSpouts(S, EnableMovingSpouts);
[S] = m_InitGUI.UpdatePassive(S, EnablePassive, PassiveSessMode);


%% Connect Maestro

if (EnableMovingSpouts == 1)
    M = PololuMaestro('COM13');
    M.setMotor(0, m_TrialConfig.ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
    M.setMotor(1, m_TrialConfig.ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));
end


%% Trial initilization

% set max number of trials

BpodSystem.Data.TrialTypes = [];
BpodSystem.Data.ProcessedSessionData = {};

% initialize anti-bias variables
AntiBiasVar.IncorrectFlag       = 0;
AntiBiasVar.IncorrectType       = 1;
AntiBiasVar.CompletedHist.left  = [];
AntiBiasVar.CompletedHist.right = [];
AntiBiasVar.BiasIndex           = 0;
AntiBiasVar.ValveFlag           = 'NoBias';

% draw perturbation interval from uniform distribution in range
PerturbInterval.EasyMinPercent       = 3/4;
PerturbInterval.EasyMaxPercent       = 1;
PerturbInterval.MediumEasyMinPercent = 1/2;
PerturbInterval.MediumEasyMaxPercent = 3/4;
PerturbInterval.MediumHardMinPercent = 1/4;
PerturbInterval.MediumHardMaxPercent = 1/2;
PerturbInterval.HardMinPercent       = 0;
PerturbInterval.HardMaxPercent       = 1/4;

% generate trial types
switch EnablePassive
    case 0
        [TrialTypes] = m_TrialConfig.GenTrials(S);
        [TrialTypes] = m_TrialConfig.AdjustWarmupTrials(S, TrialTypes);
    case 1
        [TrialTypes] = m_TrialConfig.GenPassiveTrials(S);
end

% Side Outcome Plot
BpodSystem.ProtocolFigures.OutcomePlotFig = figure('Position', [50 540 1000 220],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', TrialTypes);
BpodParameterGUI('init', S); % Initialize parameter GUI plugin


%% Define stimuli and send to analog module

SF = 44100; % Use lower sampling rate (samples/sec) to allow for longer duration audio file (max length limited by HiFi)
H.SamplingRate = SF;
Envelope = 1/(SF*0.001):1/(SF*0.001):1; % Define 1ms linear ramp envelope of amplitude coefficients, to apply at sound onset + in reverse at sound offset

H.DigitalAttenuation_dB = -35; % Set a comfortable listening level for most headphones (useful during protocol dev).


%% Setup video

BpodSystem.PluginObjects.V = [];
BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(MonitorID, 0, [0 0], [180 180], 0); % Assumes Sync patch = 180x180 pixels
BpodSystem.PluginObjects.V.SyncPatchIntensity = 255;
BpodSystem.PluginObjects.V.loadText(1, 'Loading...', '', 80);
BpodSystem.PluginObjects.V.play(1);
Xsize = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
Ysize = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
FPS   = BpodSystem.PluginObjects.V.DetectedFrameRate;

% load frame images

[VideoGrayFixed] = m_AVstimConfig.GenGreyImg(Xsize, Ysize);
BpodSystem.PluginObjects.V.loadVideo(1, VideoGrayFixed);

ImgParams.spatialFreq = 0.005;
ImgParams.contrast    = 1;
ImgParams.phase       = 0.5;

ImgParams.orientation = 0;
[VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
BpodSystem.PluginObjects.V.loadVideo(2, VideoGrating);

ImgParams.orientation = 45;
[VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
BpodSystem.PluginObjects.V.loadVideo(3, VideoGrating);

ImgParams.orientation = 90;
[VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
BpodSystem.PluginObjects.V.loadVideo(4, VideoGrating);

ImgParams.orientation = 135;
[VideoGrating] = m_AVstimConfig.GenStimImg(ImgParams, Xsize, Ysize);
BpodSystem.PluginObjects.V.loadVideo(5, VideoGrating);

VisStim.GratingIdx = [2 3 4 5];
VisStim.OmiFlag = 'False';

pause(1.0);
BpodSystem.PluginObjects.V.play(0);
BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler';
BpodSystem.PluginObjects.V.TimerMode = 2;
input('Set parameters and press enter to continue >', 's'); 
S = BpodParameterGUI('sync', S);


%% Main trial loop

for currentTrial = 1:S.GUI.MaxTrials

    S = BpodParameterGUI('sync', S);

    if (S.GUI.EnablePassive == 1 && currentTrial == 1)
        pause(S.GUI.SpontSilenceTime)
    end

    %% anti bias

    [TrialTypes] = m_TrialConfig.ManuallFraction( ...
        S, currentTrial, TrialTypes);

    [TrialTypes, AntiBiasVar] = m_TrialConfig.RepeatedIncorrect( ...
        BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes);

    [AntiBiasVar, LeftValveAmount_uL, RightValveAmount_uL, TrialTypes] = m_TrialConfig.AntiBiasValveAdjust( ...
        BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes);

    LeftValveTime   = m_TrialConfig.Amount2Time(LeftValveAmount_uL, 1);
    RightValveTime  = m_TrialConfig.Amount2Time(RightValveAmount_uL, 3);


    %% get difficulty level for this trial
    
    [TrialDifficulty] = m_TrialConfig.SamplingDiff(S);

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

    %% duration related configs

    % Draw trial-specific and difficulty-defined TimeOutPunish from exponential distribution
    DURA.TimeOutPunish = m_TrialConfig.GetTimeOutPunish(S);
    
    % Draw trial-specific ITI from exponential distribution
    DURA.ITI = m_TrialConfig.GetITI(S);

    % Draw trial-specific choice window from default or overwrite with gui
    DURA.ChoiceWindow = m_TrialConfig.GetChoiceWindow(S);

    % changing mind window
    DURA.ChangeMindDur = m_TrialConfig.GetChangeMindDur(S);
    

    %% set vis stim perturbation ISI duration according to trial-specific difficulty level

    % perturbation sampling
    if (S.GUI.EnablePassive == 0)
        [PostPertISI, EasyMaxInfo] = m_TrialConfig.GetPostPertISI( ...
            S, TrialDifficulty, PerturbInterval);
        [GrayPerturbISI] = m_TrialConfig.SetPostPertISI( ...
            S, TrialTypes, currentTrial, PostPertISI);
    else
        [PostPertISI, EasyMaxInfo] = m_TrialConfig.GetPostPertISIPassive( ...
            S, TrialTypes, currentTrial, PerturbInterval);
        [GrayPerturbISI] = m_TrialConfig.SetPostPertISIPassive( ...
            S, PostPertISI);
    end

    BpodSystem.Data.TrialVars.Trial{currentTrial}.PostPertISI = PostPertISI + S.GUI.GratingDur_s;

    m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 0);


    %% construct preperturb vis stim videos and audio stim base for grating and gray if duration parameters changed
    
    % config audio stimulus
    m_AVstimConfig.ConfigHifi(H, S, SF, Envelope);

    [JitterFlag] = m_TrialConfig.GetJitterFlag( ...
        S, TrialTypes, currentTrial, EnablePassive);

    % generate video
    VisStim.ProcessedData.Seq = [];
    VisStim.ProcessedData.PrePost = [];
    switch JitterFlag
        case 'False'
            VisStim = m_AVstimConfig.GetVideoDataPre(S, BpodSystem, FPS, VisStim, 0);
            VisStim = m_AVstimConfig.GetVideoDataPost(S, BpodSystem, FPS, VisStim, GrayPerturbISI, 0);
            VisStim = m_AVstimConfig.GetVideoDataExtra(S, BpodSystem, FPS, VisStim, GrayPerturbISI, 0);
        case 'True'
            VisStim = m_AVstimConfig.GetVideoDataPre(S, BpodSystem, FPS, VisStim, 1);
            VisStim = m_AVstimConfig.GetVideoDataPost(S, BpodSystem, FPS, VisStim, GrayPerturbISI, 1);
            VisStim = m_AVstimConfig.GetVideoDataExtra(S, BpodSystem, FPS, VisStim, GrayPerturbISI, 1);
    end

    % combine full video
    VisStim.Data.Full = m_AVstimConfig.GetFullVideo(VisStim);

    % load constructed video into the video object
    BpodSystem.PluginObjects.V.Videos{25} = struct;
    BpodSystem.PluginObjects.V.Videos{25}.nFrames = length(VisStim.Data.Full); 
    BpodSystem.PluginObjects.V.Videos{25}.Data = VisStim.Data.Full;

    VisStimDuration = VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur;


    %% Generate audio stim based on vis stim for this trial
    
    m_AVstimConfig.ConfigFullAudioStim( ...
        H, S, VisStim, SF, Envelope);
    
    
    %% trial target
    % port1:left, port2:center, port3:right
    
    TargetConfig.Left.CorrectChoice = 'Left';
    TargetConfig.Left.CorrectLick   = 'Port1In';
    TargetConfig.Left.CorrectPort   = 'Port1';
    TargetConfig.Left.IncorrectLick = 'Port3In';
    TargetConfig.Left.IncorrectPort = 'Port3';
    TargetConfig.Left.Valve         = 'Valve1';
    TargetConfig.Left.ValveTime     = LeftValveTime;
    TargetConfig.Right.CorrectChoice = 'Right';
    TargetConfig.Right.CorrectLick   = 'Port3In';
    TargetConfig.Right.CorrectPort   = 'Port3';
    TargetConfig.Right.IncorrectLick = 'Port1In';
    TargetConfig.Right.IncorrectPort = 'Port1';
    TargetConfig.Right.Valve         = 'Valve3';
    TargetConfig.Right.ValveTime     = RightValveTime;
    switch TrialTypes(currentTrial)
        case 1
            TrialTarget = TargetConfig.Left;
        case 2
            TrialTarget = TargetConfig.Right;
    end


    %% set softcode for hardware control output actions

    SCOA.Start       = {'HiFi1','*', 'BNC1', 1};
    SCOA.InitCue     = {'HiFi1', ['P' 0]};
    SCOA.StimAct     = m_AVstimConfig.GetStimAct(S);
    SCOA.EarlyChoice = {'SoftCode', 255, 'HiFi1', 'X'};
    SCOA.Punish      = {'SoftCode', 255, 'HiFi1', ['P' 2]};
    SCOA.VisStim     = {'SoftCode', 25};
    % SCOA.AudStim     = {'HiFi1', ['P', 4], 'BNCState', 1, 'BNC1', 1};
    SCOA.AudStim     = {'HiFi1', ['P', 4]};
    if (EnableMovingSpouts == 1)
        SCOA.SpoutIn = {'SoftCode', 9};
    else
        SCOA.SpoutIn = {};
    end


    %% construct state matrix

    switch S.GUI.TrainingLevel
        case 1 % passive
            ExperimenterTrialInfo.TrainingLevel = 'Passive';
            StatePassive(S, SCOA, VisStimDuration, DURA);
        case 2 % naive
            ExperimenterTrialInfo.TrainingLevel = 'Naive';
            StateNaive(S, SCOA, TrialTarget, VisStimDuration, DURA);
        case 3 % Mid 1 Trained
            if (currentTrial <= S.GUI.NumNaiveWarmup)
                ExperimenterTrialInfo.TrainingLevel = 'Naive warmup';
                StateNaive(S, SCOA, TrialTarget, VisStimDuration, DURA);
            else
                ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 1';
                StateMidTrain1(S, SCOA, TrialTarget, VisStimDuration, DURA);
            end
        case 4 % Mid 2 Trained
            if (currentTrial <= S.GUI.NumNaiveWarmup)
                ExperimenterTrialInfo.TrainingLevel = 'Naive warmup';
                StateNaive(S, SCOA, TrialTarget, VisStimDuration, DURA);
            else
                ExperimenterTrialInfo.TrainingLevel = 'Mid Trained 2';
                StateMidTrain2(S, SCOA, TrialTarget, VisStimDuration, DURA);
            end
        case 5 % well trained
            if (currentTrial <= S.GUI.NumNaiveWarmup)
                ExperimenterTrialInfo.TrainingLevel = 'Naive warmup';
                StateNaive(S, SCOA, TrialTarget, VisStimDuration, DURA);
            else
                ExperimenterTrialInfo.TrainingLevel = 'Well Trained';
                StateWellTrained(S, SCOA, TrialTarget, VisStimDuration, DURA);
            end
    end


    %% add console print for experimenter trial information

    ExperimenterTrialInfo.TrialNumber = currentTrial;
    ExperimenterTrialInfo.CorrectChoice = TrialTarget.CorrectChoice;
    ExperimenterTrialInfo.Bias = AntiBiasVar.ValveFlag;
    ExperimenterTrialInfo.LeftValveAmount  = LeftValveAmount_uL;
    ExperimenterTrialInfo.RightValveAmount = RightValveAmount_uL;
    ExperimenterTrialInfo.PostPertISI = GrayPerturbISI;
    ExperimenterTrialInfo.VisStimDuration = VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur + VisStim.Data.Extra.Dur;
    ExperimenterTrialInfo.CategoryBoundary = S.GUI.ISIOrig_s;
    ExperimenterTrialInfo.EasyMax = EasyMaxInfo;
    ExperimenterTrialInfo.ITI = DURA.ITI;
    ExperimenterTrialInfo.TimeOutPunish = DURA.TimeOutPunish;
    ExperimenterTrialInfo.ChoiceWindow = DURA.ChoiceWindow;
    ExperimenterTrialInfo.ChangeMindDur = DURA.ChangeMindDur;
    ExperimenterTrialInfo.PreISIinfo = VisStim.PreISIinfo;
    ExperimenterTrialInfo.PostISIinfo = VisStim.PostISIinfo;
    ExperimenterTrialInfo.Jitter = JitterFlag;
    ExperimenterTrialInfo.Omission = VisStim.OmiFlag;

    strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
    disp(strExperimenterTrialInfo);

    RawEvents = RunStateMachine;

    if (S.GUI.EnablePassive == 1)
        m_TrialConfig.PassiveBlockSleep(S, TrialTypes, currentTrial)
    end
    

    %% save data and update plot

    if ~isempty(fieldnames(RawEvents))
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents);
        BpodSystem.Data.TrialSettings(currentTrial) = S;
        BpodSystem.Data.TrialTypes(currentTrial) = TrialTypes(currentTrial);
        m_PostProcess.SaveProcessedSessionData(BpodSystem, VisStim, GrayPerturbISI);
        m_Plotter.UpdateOutcomePlot(BpodSystem, TrialTypes, 1);
        StateTiming();
        SaveBpodSessionData;
    end
    HandlePauseCondition;
    if BpodSystem.Status.BeingUsed == 0
        clear global M;
        BpodSystem.PluginObjects.V = [];
        BpodSystem.setStatusLED(1);
        return
    end

end

clear global M;
BpodSystem.PluginObjects.V = [];
BpodSystem.setStatusLED(1);
