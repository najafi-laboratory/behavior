function block_single_interval_discrimination_V_1
protocolVersion = 'single_interval_discrimination_V_1_11';

ctx = struct();
try
    ctx = setupProtocol();
    ctx.protocolVersion = protocolVersion;
    ctx = waitForExperimenterSettings(ctx);
    runTrials(ctx);
    cleanupProtocol(ctx);
catch err
    disp(err.identifier);
    disp(getReport(err));
    writeCrashLog(err);
    cleanupProtocol(ctx);
end
end

function ctx = setupProtocol()
global BpodSystem AntiBiasVar M EnLaser S

ctx.enableMovingSpouts = 1;
ctx.enableOpto = 0;
EnLaser = ctx.enableOpto;

ctx.plotter = Plotter;
ctx.gui = InitGUI;
ctx.trial = TrialConfig;
ctx.av = AVstimConfig;
ctx.post = PostProcess;
ctx.opto = OptoConfig;

BpodSystem.Data.PlotCntr = 1;
BpodSystem.Data.CurrentTrial = 1;
setRigID();

BpodSystem.setStatusLED(0);
BpodSystem.assertModule('HiFi', 1);
ctx.hifi = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);

S = ctx.gui.SetParams(BpodSystem);
S = ctx.gui.UpdateMovingSpouts(S, ctx.enableMovingSpouts);

M = connectMaestro(S, ctx.trial, ctx.enableMovingSpouts);
AntiBiasVar = initAntiBias();
BpodSystem.Data = initSessionData(BpodSystem.Data);
ctx.perturbInterval = initPerturbInterval();

[ctx.trialTypes, ctx.blockTypes] = ctx.trial.GenTrials(S);
ctx.trialTypes = ctx.trial.AdjustWarmupTrials(S, ctx.trialTypes);
ctx.probeTrials = ctx.trial.GenProbeTrials(S);
ctx.optoType = ctx.opto.GenOptoType(S, ctx.blockTypes);

if EnLaser
    ctx.opto = ctx.opto.ConnectInitLaser(S);
end

initOutcomePlot(ctx.plotter, ctx.trialTypes, ctx.optoType, ctx.probeTrials, S);
[ctx.hifi, ctx.sampleRate, ctx.envelope] = configureHifi(ctx.hifi);
[ctx.visStim, ctx.frameRate] = configureVideo(ctx.av, getMonitorID(), S);
end

function ctx = waitForExperimenterSettings(ctx)
global BpodSystem S

BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler';
BpodSystem.PluginObjects.V.play(6);
input('Set parameters and press enter to continue >', 's');
S = BpodParameterGUI('sync', S);

if S.GUI.ActMaxSameSide
    ctx.trialTypes = ctx.trial.AdjustMaxConsecutiveSameSideTrials(S, ctx.trialTypes);
end
ctx.probeTrials = ctx.trial.GenProbeTrials(S);
ctx.optoType = ctx.opto.GenOptoType(S, ctx.blockTypes);
ctx.plotter.UpdateOutcomePlot(BpodSystem, ctx.trialTypes, ctx.optoType, ctx.probeTrials, 0);
end

function runTrials(ctx)
global BpodSystem AntiBiasVar TargetConfig S

for currentTrial = 1:S.GUI.MaxTrials
    BpodSystem.Data.CurrentTrial = currentTrial;
    S = BpodParameterGUI('sync', S);

    [ctx.trialTypes, AntiBiasVar, valve] = configureBias(ctx.trial, currentTrial, ctx.trialTypes);
    saveTrialSchedule(currentTrial, ctx.trialTypes, ctx.blockTypes, AntiBiasVar);

    trialDifficulty = ctx.trial.SamplingDiff(S);
    dura = getDurations(ctx.trial);
    [prePertISI, postPertISI, easyMaxInfo] = ctx.trial.GetPostPertISI( ...
        S, trialDifficulty, ctx.perturbInterval, ctx.trialTypes, currentTrial);
    saveTrialISIs(currentTrial, prePertISI, postPertISI);

    jitterFlag = ctx.trial.GetJitterFlag(S);
    ctx.av.ConfigHifi(ctx.hifi, S, ctx.sampleRate, ctx.envelope);
    ctx.visStim = buildTrialStimulus(ctx.av, ctx.visStim, ctx.frameRate, prePertISI, postPertISI, jitterFlag);
    ctx.hifi.load(5, ctx.av.GenAudioStim(S, ctx.visStim, ctx.sampleRate, ctx.envelope));

    TargetConfig = getTargetConfig(ctx.trial, valve.left_uL, valve.right_uL);
    trialTarget = selectTarget(TargetConfig, ctx.trialTypes(currentTrial));
    scoa = getSoftCodeActions(ctx.opto, ctx.optoType, currentTrial, ctx.enableMovingSpouts);

    info = buildAndSendStateMachine(ctx.opto, ctx.optoType, currentTrial, scoa, trialTarget, ctx.visStim, dura);
    info = addTrialInfo(info, currentTrial, trialDifficulty, ctx.visStim, trialTarget, AntiBiasVar, valve, postPertISI, dura, easyMaxInfo, jitterFlag);
    disp(formattedDisplayText(info, 'UseTrueFalseForLogical', true));

    rawEvents = RunStateMachine;
    if ~isempty(fieldnames(rawEvents))
        saveCompletedTrial(ctx, currentTrial, rawEvents, postPertISI, jitterFlag);
    end

    HandlePauseCondition;
    if BpodSystem.Status.BeingUsed == 0
        printSessionSummary(ctx.protocolVersion, currentTrial, info);
        return
    end
end
end

function [trialTypes, antiBias, valve] = configureBias(trialConfig, currentTrial, trialTypes)
global BpodSystem AntiBiasVar S

[trialTypes, antiBias] = trialConfig.RepeatedIncorrect(BpodSystem, S, AntiBiasVar, currentTrial, trialTypes);
[antiBias, valve.left_uL, valve.right_uL, trialTypes] = trialConfig.AntiBiasValveAdjust( ...
    BpodSystem, S, antiBias, currentTrial, trialTypes);

if currentTrial > 1 && S.GUI.TrainingLevel == 1 && trialHadState(currentTrial - 1, 'PunishNaive')
    trialTypes(currentTrial) = trialTypes(currentTrial - 1);
end

[antiBias, valve.left_uL, valve.right_uL] = trialConfig.AntiBiasProbeTrials( ...
    BpodSystem, S, antiBias, currentTrial, trialTypes, valve.left_uL, valve.right_uL);
antiBias = trialConfig.ManualSingleSpout(BpodSystem, S, antiBias);
end

function saveTrialSchedule(currentTrial, trialTypes, blockTypes, antiBias)
global BpodSystem
BpodSystem.Data.TrialTypes(currentTrial) = trialTypes(currentTrial);
BpodSystem.Data.BlockTypes(currentTrial) = blockTypes(currentTrial);
BpodSystem.Data.IsAntiBiasProbeTrial(currentTrial) = antiBias.IsAntiBiasProbeTrial;
BpodSystem.Data.MoveCorrectSpout(currentTrial) = antiBias.MoveCorrectSpout;
end

function saveTrialISIs(currentTrial, prePertISI, postPertISI)
global BpodSystem S
BpodSystem.Data.TrialVars.Trial{currentTrial}.PostPertISI = postPertISI + S.GUI.GratingDur_s;
BpodSystem.Data.TrialVars.Trial{currentTrial}.PostPertISIWithoutGrating = postPertISI;
BpodSystem.Data.TrialVars.Trial{currentTrial}.PrePertISI = prePertISI + S.GUI.GratingDur_s;
BpodSystem.Data.TrialVars.Trial{currentTrial}.PrePertISIWithoutGrating = prePertISI;
end

function dura = getDurations(trialConfig)
global S
dura.TimeOutPunish = trialConfig.GetTimeOutPunish(S);
dura.ITI = trialConfig.GetITI(S);
dura.ChoiceWindow = trialConfig.GetChoiceWindow(S);
dura.ChangeMindDur = trialConfig.GetChangeMindDur(S);
dura.PostVisStimDelay = trialConfig.GetPostVisStimDelay(S);
end

function visStim = buildTrialStimulus(avConfig, visStim, frameRate, prePertISI, postPertISI, jitterFlag)
global BpodSystem S
visStim.ProcessedData.Seq = [];
visStim.ProcessedData.PrePost = [];
visStim = avConfig.GetVisStimImg(S, BpodSystem, frameRate, visStim);
visStim = avConfig.GetVideoDataPre(S, BpodSystem, frameRate, visStim, prePertISI, jitterFlag);
visStim = avConfig.GetVideoDataPost(S, BpodSystem, frameRate, visStim, postPertISI, jitterFlag);
visStim.Data.Full = avConfig.GetFullVideo(S, visStim, frameRate);
visStim.Data.VisStimDuration = visStim.Data.Pre.Dur + visStim.Data.Post.Dur;
BpodSystem.PluginObjects.V.Videos{25} = struct('nFrames', length(visStim.Data.Full), 'Data', visStim.Data.Full);
end

function targetConfig = getTargetConfig(trialConfig, leftAmount_uL, rightAmount_uL)
targetConfig.Left = struct( ...
    'CorrectChoice', 'Left', ...
    'CorrectLick', 'Port1In', ...
    'CorrectPort', 'Port1', ...
    'IncorrectLick', 'Port3In', ...
    'IncorrectPort', 'Port3', ...
    'Valve', 'Valve1', ...
    'ValveTime', trialConfig.Amount2Time(leftAmount_uL, 1));
targetConfig.Right = struct( ...
    'CorrectChoice', 'Right', ...
    'CorrectLick', 'Port3In', ...
    'CorrectPort', 'Port3', ...
    'IncorrectLick', 'Port1In', ...
    'IncorrectPort', 'Port1', ...
    'Valve', 'Valve3', ...
    'ValveTime', trialConfig.Amount2Time(rightAmount_uL, 3));
end

function trialTarget = selectTarget(targetConfig, trialType)
if trialType == 1
    trialTarget = targetConfig.Left;
else
    trialTarget = targetConfig.Right;
end
end

function scoa = getSoftCodeActions(optoConfig, optoType, currentTrial, enableMovingSpouts)
global S
scoa.Start = {'HiFi1', '*', 'BNC1', 1};
scoa.InitCue = {'HiFi1', ['P' 0]};
scoa.StimAct = {};
scoa.Punish = {'HiFi1', ['P' 2]};
scoa.VisStim = {};
scoa.SpoutIn = {};
if S.GUI.VisStimEnable == 1
    scoa.VisStim = {'SoftCode', 25};
end
if enableMovingSpouts == 1
    scoa.SpoutIn = {'SoftCode', 9};
end
scoa.AudStim = optoConfig.OptoUpdateAudStimTrig(optoType, currentTrial);
scoa.ITI = optoConfig.GetITIAction(optoType, currentTrial);
end

function info = buildAndSendStateMachine(optoConfig, optoType, currentTrial, scoa, trialTarget, visStim, dura)
global BpodSystem S
sma = NewStateMatrix();
optoDuration = optoConfig.GetOptoDuration(S, visStim, dura, optoType, currentTrial);
sma = optoConfig.SetOpto(BpodSystem, S, sma, optoDuration, optoType, currentTrial);
info.Opto = optoLabel(optoType(currentTrial));

if currentTrial <= S.GUI.NumNaiveWarmup && S.GUI.TrainingLevel > 1
    info.TrainingLevel = 'Naive warmup';
    StateNaive(sma, S, scoa, trialTarget, visStim.Data.VisStimDuration, dura);
    return
end

switch S.GUI.TrainingLevel
    case 1
        info.TrainingLevel = 'Naive';
        StateNaive(sma, S, scoa, trialTarget, visStim.Data.VisStimDuration, dura);
    case 2
        info.TrainingLevel = 'Early';
        StateEarlyTrain(sma, S, scoa, trialTarget, visStim.Data.VisStimDuration, dura);
    case 3
        info.TrainingLevel = 'Mid 1';
        StateMidTrain(sma, S, scoa, trialTarget, visStim.Data.VisStimDuration, dura);
    case 4
        info.TrainingLevel = 'Mid 2';
        StateMidTrain(sma, S, scoa, trialTarget, visStim.Data.VisStimDuration, dura);
    case 5
        info.TrainingLevel = 'Well';
        StateWellTrained(sma, S, scoa, trialTarget, visStim.Data.VisStimDuration, dura);
end
end

function label = optoLabel(optoType)
labels = {'off', 'type 1', 'type 2'};
label = labels{optoType + 1};
end

function info = addTrialInfo(info, currentTrial, difficulty, visStim, trialTarget, antiBias, valve, postPertISI, dura, easyMaxInfo, jitterFlag)
global S
info.TrialNumber = currentTrial;
info.Difficulty = difficultyLabel(difficulty);
info.GratingOrientation = visStim.orientation(visStim.SampleGratingIdx - 1);
info.CorrectChoice = trialTarget.CorrectChoice;
info.ReactionTask = S.GUI.ReactionTask;
info.Bias = antiBias.ValveFlag;
info.LeftValveAmount = valve.left_uL;
info.RightValveAmount = valve.right_uL;
info.PostPertISI = postPertISI;
info.VisStimDuration = visStim.Data.Post.Dur;
info.CategoryBoundary = S.GUI.ISIShortMax_s;
info.EasyMax = easyMaxInfo;
info.ITI = dura.ITI;
info.TimeOutPunish = dura.TimeOutPunish;
info.ChoiceWindow = dura.ChoiceWindow;
info.PostISIinfo = visStim.PostISIinfo;
info.Jitter = jitterFlag;
info.Omission = visStim.OmiFlag;
end

function label = difficultyLabel(difficulty)
labels = {'Easy', 'Medium-Easy', 'Medium-Hard', 'Hard'};
label = labels{difficulty};
end

function saveCompletedTrial(ctx, currentTrial, rawEvents, postPertISI, jitterFlag)
global BpodSystem S
BpodSystem.Data = AddTrialEvents(BpodSystem.Data, rawEvents);
BpodSystem.Data.TrialSettings(currentTrial) = S;
BpodSystem.Data.TrialTypes(currentTrial) = ctx.trialTypes(currentTrial);
BpodSystem.Data.OptoType(currentTrial) = ctx.optoType(currentTrial);
BpodSystem.Data.JitterFlag(currentTrial) = jitterFlag;
BpodSystem = ctx.post.SaveProcessedSessionData(BpodSystem, ctx.visStim, postPertISI);
ctx.plotter.UpdateOutcomePlot(BpodSystem, ctx.trialTypes, ctx.optoType, ctx.probeTrials, 1);
SavePlot;
StateTiming();
SaveBpodSessionData;
end

function data = initSessionData(data)
data.TrialTypes = [];
data.ProbeTrials = [];
data.IsAntiBiasProbeTrial = [];
data.MoveCorrectSpout = [];
data.OptoType = [];
data.ProcessedSessionData = {};
end

function antiBias = initAntiBias()
antiBias.IncorrectFlag = 0;
antiBias.IncorrectType = 1;
antiBias.CompletedHist.left = [];
antiBias.CompletedHist.right = [];
antiBias.BiasIndex = 0;
antiBias.ValveFlag = 'NoBias';
antiBias.ServoRightAdjust = 0;
antiBias.ServoRightTrialsSinceAdjust = 20;
antiBias.ServoLeftAdjust = 0;
antiBias.ServoLeftTrialsSinceAdjust = 20;
antiBias.IsAntiBiasProbeTrial = false;
antiBias.AutoMoveSpout = false;
antiBias.MoveCorrectSpout = false;
antiBias.NumSpoutSelectTrials = 3;
antiBias.NumProbeTrials = 10;
end

function perturbInterval = initPerturbInterval()
perturbInterval.EasyMinPercent = 3/4;
perturbInterval.EasyMaxPercent = 1;
perturbInterval.MediumEasyMinPercent = 1/2;
perturbInterval.MediumEasyMaxPercent = 3/4;
perturbInterval.MediumHardMinPercent = 1/4;
perturbInterval.MediumHardMaxPercent = 1/2;
perturbInterval.HardMinPercent = 0;
perturbInterval.HardMaxPercent = 1/4;
end

function maestro = connectMaestro(S, trialConfig, enabled)
global BpodSystem
maestro = [];
if ~enabled
    return
end

switch BpodSystem.Data.RigName
    case 'ImagingRig'
        port = 'COM15';
    case '2AFCRig1'
        port = 'COM16';
    case '2AFCRig2'
        port = 'COM5';
    otherwise
        error('No Maestro port configured for rig "%s".', BpodSystem.Data.RigName);
end

maestro = PololuMaestro(port);
maestro.setMotor(0, trialConfig.ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
maestro.setMotor(1, trialConfig.ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));
end

function initOutcomePlot(plotter, trialTypes, optoType, probeTrials, S)
global BpodSystem
BpodSystem.ProtocolFigures.OutcomePlotFig = figure( ...
    'Position', [50 540 1000 220], ...
    'name', 'Outcome plot', ...
    'numbertitle', 'off', ...
    'MenuBar', 'none', ...
    'Resize', 'off');
BpodSystem.GUIHandles.OutcomePlot = axes('Position', [.075 .35 .89 .55]);
TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot, 'init', trialTypes, optoType, probeTrials);
BpodParameterGUI('init', S);
plotter.UpdateOutcomePlot(BpodSystem, trialTypes, optoType, probeTrials, 0);
set(BpodSystem.ProtocolFigures.OutcomePlotFig, 'Position', [917 829 1000 220]);
set(BpodSystem.ProtocolFigures.ParameterGUI, 'Position', [8 51 2604 992]);
end

function [hifi, sampleRate, envelope] = configureHifi(hifi)
sampleRate = 44100;
hifi.SamplingRate = sampleRate;
envelope = 1/(sampleRate*0.001):1/(sampleRate*0.001):1;
hifi.DigitalAttenuation_dB = -35;
end

function [visStim, frameRate] = configureVideo(avConfig, monitorID, S)
global BpodSystem
BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(monitorID, 0, [0 0], [180 180], 0);
BpodSystem.PluginObjects.V.SyncPatchIntensity = 255;
BpodSystem.PluginObjects.V.loadText(1, 'Loading...', '', 80);
BpodSystem.PluginObjects.V.play(1);

xSize = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
ySize = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
frameRate = BpodSystem.PluginObjects.V.DetectedFrameRate;
BpodSystem.PluginObjects.V.loadVideo(1, avConfig.GenGreyImg(xSize, ySize));

img.spatialFreq = 0.005;
img.contrast = 1;
img.phase = 0.5;
visStim.orientation = [0 45 90 135];
visStim.GratingIdx = 2:5;
visStim.OmiFlag = 'False';
for i = 1:numel(visStim.orientation)
    img.orientation = visStim.orientation(i);
    BpodSystem.PluginObjects.V.loadVideo(visStim.GratingIdx(i), avConfig.GenStimImg(img, xSize, ySize));
end

visStim = avConfig.GetVisStimImg(S, BpodSystem, frameRate, visStim);
grayInitBNCSync = [repmat(visStim.Img.GrayFrame_SyncW, 1, 120) visStim.Img.GrayFrame_SyncBlk];
BpodSystem.PluginObjects.V.Videos{6} = struct('nFrames', 121, 'Data', grayInitBNCSync);
pause(1.0);
BpodSystem.PluginObjects.V.TimerMode = 0;
BpodSystem.PluginObjects.V.play(0);
end

function monitorID = getMonitorID()
global BpodSystem
switch BpodSystem.Data.RigName
    case {'ImagingRig', '2AFCRig1', '2AFCRig2'}
        monitorID = 1;
    otherwise
        monitorID = 1;
end
end

function stateHit = trialHadState(trialIndex, stateName)
global BpodSystem
states = BpodSystem.Data.RawEvents.Trial{trialIndex}.States;
stateHit = isfield(states, stateName) && ~isnan(states.(stateName)(1));
end

function printSessionSummary(protocolVersion, currentTrial, info)
global S
fprintf('\n\n');
fprintf('%s\n', char(datetime('now', 'Format', 'MM/dd/yy')));
fprintf('%s\n\n', char(datetime('now', 'Format', 'eeee')));
fprintf('%s\n\n', char(datetime('now', 'Format', 'h:mm a')));
fprintf('%s\n', S.GUI.ExperimenterInitials);
fprintf('Protocol Version: %s\n', protocolVersion);
fprintf('%s\n', info.TrainingLevel);
fprintf('Total Trials: %s\n\n\n', num2str(currentTrial));
end

function cleanupProtocol(ctx)
global BpodSystem M AntiBiasVar EnLaser S
if isfield(ctx, 'trial') && ~isempty(M) && ~isempty(S)
    try
        M.setMotor(0, ctx.trial.ConvertMaestroPos(S.GUI.RightServoInPos));
        M.setMotor(1, ctx.trial.ConvertMaestroPos(S.GUI.LeftServoInPos));
    catch cleanupErr
        disp(cleanupErr.message);
    end
end
M = [];
AntiBiasVar = [];

try
    BpodSystem.PluginObjects.V = [];
    BpodSystem.setStatusLED(1);
catch cleanupErr
    disp(cleanupErr.message);
end

if EnLaser && isfield(ctx, 'opto')
    try
        ctx.opto.DisconnectLaser(S);
    catch cleanupErr
        disp(cleanupErr.message);
    end
end
end

function writeCrashLog(err)
global BpodSystem
if isempty(BpodSystem) || ~isfield(BpodSystem, 'Path') || ~isfield(BpodSystem.Path, 'CurrentDataFile')
    return
end

t = datetime;
sessionDate = 10000*(year(t)-2000) + 100*month(t) + day(t);
[~, sessionFileName] = fileparts(BpodSystem.Path.CurrentDataFile);
crashDir = 'C:\data analysis\behavior\error logs\';
crashFile = [crashDir, num2str(sessionDate), '_BPod-matlab_crash_log_', sessionFileName];
mkdir(crashDir);

Data = BpodSystem.Data;
save(crashFile, 'Data');

fid = fopen([crashFile, '.txt'], 'a+');
if fid == -1
    return
end
fprintf(fid, '%s\n', sessionFileName);
fprintf(fid, '%s\n', num2str(sessionDate));
fprintf(fid, '%s\n', BpodSystem.Data.RigName);
fprintf(fid, '%s\n', err.identifier);
fprintf(fid, '%s\n', err.message);
fprintf(fid, '%s\n', err.Correction);
fprintf(fid, '%s', err.getReport('extended', 'hyperlinks', 'off'));
fclose(fid);
end

function setRigID()
global BpodSystem
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
    case 'COS-3A11264'
        BpodSystem.Data.RigName = '2AFCRig2';
    case 'COS-3A11215'
        BpodSystem.Data.RigName = '2AFCRig1';
end
end