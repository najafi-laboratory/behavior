function block_single_interval_discrimination_202606
global BpodSystem
global S
global M

cleanupHandle = onCleanup(@cleanupProtocol);

removeProtocolSubfoldersFromPath;
S = ConfigureProtocol(BpodSystem);
BpodParameterGUI('init', S);
positionParameterGUI;
input('Set parameters in the GUI and press Enter to configure the session > ', 's');
S = BpodParameterGUI('sync', S);
validateSettings(S);
confirmDoricOptoSettings(S);
initializeSessionData;

initializeHiFi(S);
initializeVideo(S);
holdReadyGrayScreen;
initializeServo(S);
waitForGrayScreenEnter;
BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_BlockSingleInterval';

[trialTypes, blockTypes, blockStarts, blockEnds] = GenerateTrials(S);
isiValues = nan(1, numel(trialTypes));
itiValues = nan(1, numel(trialTypes));
punishITIValues = nan(1, numel(trialTypes));
optoTypes = GenerateOptoTrials(S, blockTypes, blockStarts, blockEnds);
probeTypes = GenerateProbeTrials(S, blockTypes);
BpodSystem.Data.PlannedTrialTypes = trialTypes;
BpodSystem.Data.PlannedBlockTypes = blockTypes;
BpodSystem.Data.PlannedBlockStarts = blockStarts;
BpodSystem.Data.PlannedBlockEnds = blockEnds;
BpodSystem.Data.PlannedProbeTrialTypes = probeTypes;
BpodSystem.Data.PlannedOptoTrialTypes = optoTypes;
BpodSystem.Data.PlannedISI = isiValues;
BpodSystem.Data.PlannedITI = itiValues;
BpodSystem.Data.PlannedPunishITI = punishITIValues;
BpodSystem.Data.PlannedChemoTrialTypes = repmat(double(S.GUI.ChemoMode), 1, numel(trialTypes));
currentTrial = 1;
BpodSystem.Data.CurrentTrial = 0;
BpodSystem.Data.AssignedOptoTrialCount = 0;

ProtocolPlot('init', trialTypes, blockTypes, probeTypes, optoTypes, isiValues, 0, S);
showGrayScreen;

while currentTrial <= round(S.GUI.MaxTrials)
    BpodSystem.Data.CurrentTrial = currentTrial;
    S = BpodParameterGUI('sync', S);
    validateSettings(S);
    maxTrials = round(S.GUI.MaxTrials);
    if currentTrial > maxTrials
        break
    end

    trialType = trialTypes(currentTrial);
    blockType = blockTypes(currentTrial);
    probeType = probeTypes(currentTrial);
    optoType = GenerateOptoTrial(S, blockTypes, blockStarts, blockEnds, currentTrial);
    optoTypes(:, currentTrial) = optoType;
    BpodSystem.Data.PlannedOptoTrialTypes(:, currentTrial) = optoType;
    BpodSystem.Data.AssignedOptoTrialCount = currentTrial;
    isi = sampleTrialISI(S, trialType);
    isiValues(currentTrial) = isi;
    BpodSystem.Data.PlannedISI(currentTrial) = isi;
    iti = sampleTrialITI(S);
    punishITI = sampleTrialPunishITI(S);
    itiValues(currentTrial) = iti;
    punishITIValues(currentTrial) = punishITI;
    BpodSystem.Data.PlannedITI(currentTrial) = iti;
    BpodSystem.Data.PlannedPunishITI(currentTrial) = punishITI;
    target = trialTarget(S, trialType);
    stimulus = BuildStimulus(S, isi);
    loadTrialStimulus(stimulus);

    printTrialInfo(currentTrial, trialType, blockType, probeType, optoType, target, isi, iti, punishITI, stimulus);
    ProtocolPlot('update', trialTypes, blockTypes, probeTypes, optoTypes, isiValues, currentTrial - 1, S);
    showGrayScreen;

    sma = BuildStateMachine(S, stimulus.Duration_s, iti, punishITI, target, probeType, optoType);
    SendStateMachine(sma);
    rawEvents = RunStateMachine;
    if isempty(fieldnames(rawEvents))
        break
    end

    BpodSystem.Data = AddTrialEvents(BpodSystem.Data, rawEvents);
    storedEvents = BpodSystem.Data.RawEvents.Trial{currentTrial};
    BpodSystem.Data.TrialSettings(currentTrial) = S;
    BpodSystem.Data.TrialTypes(currentTrial) = trialType;
    BpodSystem.Data.BlockTypes(currentTrial) = blockType;
    BpodSystem.Data.ProbeTrialTypes(currentTrial) = probeType;
    BpodSystem.Data.OptoTrialTypes(:, currentTrial) = optoType;
    BpodSystem.Data.ChemoTrialTypes(currentTrial) = double(S.GUI.ChemoMode);
    BpodSystem.Data.ChemoTrialType(currentTrial) = double(S.GUI.ChemoMode);
    if any(BpodSystem.Data.ChemoTrialTypes(1:currentTrial) == 1)
        BpodSystem.Data.ChemoTrialTypes(1:currentTrial) = 1;
        BpodSystem.Data.ChemoTrialType(1:currentTrial) = 1;
    end
    BpodSystem.Data.ISI(currentTrial) = isi;
    BpodSystem.Data.ITI(currentTrial) = iti;
    BpodSystem.Data.PunishITI(currentTrial) = punishITI;
    BpodSystem.Data.CorrectSide{currentTrial} = target.Side;
    BpodSystem.Data.Contingency(currentTrial) = S.GUI.Contingency;
    BpodSystem.Data.Outcomes(currentTrial) = trialOutcome(storedEvents, target, probeType);
    BpodSystem.Data.StimulusDuration(currentTrial) = stimulus.Duration_s;
    BpodSystem.Data.StimulusMode(currentTrial) = S.GUI.StimulusMode;
    BpodSystem.Data.UseSavedImage(currentTrial) = stimulus.UseSavedImage;
    SaveBpodSessionData;

    ProtocolPlot('update', trialTypes, blockTypes, probeTypes, optoTypes, isiValues, currentTrial, S);
    showGrayScreen;

    if BpodSystem.Status.BeingUsed == 0
        break
    end
    HandlePauseCondition;
    if BpodSystem.Status.BeingUsed == 0
        break
    end

    currentTrial = currentTrial + 1;
end

showGrayScreen;
printSessionSummary(currentTrial - 1, S);
end

function initializeSessionData
global BpodSystem

if ~isfield(BpodSystem, 'Data') || ~isstruct(BpodSystem.Data)
    BpodSystem.Data = struct;
end
if ~isfield(BpodSystem.Data, 'CurrentTrial') || isempty(BpodSystem.Data.CurrentTrial)
    BpodSystem.Data.CurrentTrial = 0;
end
if ~isfield(BpodSystem.Data, 'RigName') || isempty(BpodSystem.Data.RigName)
    BpodSystem.Data.RigName = 'UnknownRig';
end
end

function initializeHiFi(S)
global BpodSystem

BpodSystem.assertModule('HiFi', 1);
if isfield(BpodSystem.PluginObjects, 'H') && ~isempty(BpodSystem.PluginObjects.H)
    try
        BpodSystem.PluginObjects.H.stop;
    catch
    end
    try
        delete(BpodSystem.PluginObjects.H);
    catch
    end
    BpodSystem.PluginObjects.H = [];
end
releaseSerialPort(BpodSystem.ModuleUSB.HiFi1);
pause(0.2);
BpodSystem.PluginObjects.H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);
BpodSystem.PluginObjects.H.SamplingRate = S.GUI.AudioSamplingRate_Hz;
BpodSystem.PluginObjects.H.DigitalAttenuation_dB = S.GUI.AudioAttenuation_dB;
end

function releaseSerialPort(portName)
if exist('serialportfind', 'file') ~= 2
    return
end

try
    ports = serialportfind('Port', portName);
    for i = 1:numel(ports)
        delete(ports(i));
    end
catch
end
end

function initializeVideo(~)
global BpodSystem

if isfield(BpodSystem.PluginObjects, 'V') && ~isempty(BpodSystem.PluginObjects.V)
    try
        delete(BpodSystem.PluginObjects.V);
    catch
    end
end

try
    Screen('Preference', 'VisualDebugLevel', 0);
    Screen('Preference', 'SuppressAllWarnings', 1);
catch
end
BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(1, 0, [0 0], [180 180], 0);
BpodSystem.PluginObjects.V.SyncPatchIntensity = 255;
BpodSystem.PluginObjects.V.TimerMode = 2;
BpodSystem.Data.VideoFrameRate = BpodSystem.PluginObjects.V.DetectedFrameRate;
BpodSystem.Data.VideoViewPort = BpodSystem.PluginObjects.V.ViewPortDimensions;
loadReadyGrayVideo;
prepareReadyGrayScreen;
pause(1.0);
holdReadyGrayScreen;
end

function initializeServo(S)
global BpodSystem
global M

if ~isempty(M)
    try
        delete(M);
    catch
    end
    pause(0.2);
end
maestroPort = findMaestroPort;
M = PololuMaestro(maestroPort);
BpodSystem.Data.HardwarePorts.Maestro = maestroPort;
moveSpoutsOut(S);
end

function loadTrialStimulus(stimulus)
global BpodSystem

BpodSystem.PluginObjects.V.Videos{25} = struct('nFrames', numel(stimulus.VisualVideo), 'Data', stimulus.VisualVideo);
BpodSystem.PluginObjects.V.Videos{26} = struct('nFrames', numel(stimulus.SyncVideo), 'Data', stimulus.SyncVideo);
BpodSystem.PluginObjects.H.load(5, stimulus.Audio);
end

function showGrayScreen
forceGrayScreen;
end

function waitForGrayScreenEnter
global BpodSystem

loadReadyGrayVideo;
SoftCodeHandler_BlockSingleInterval(254);
pause(1);
BpodSystem.PluginObjects.V.play(0);
SoftCodeHandler_BlockSingleInterval(254);
pause(0.2);
disp('Screen is gray and hardware is ready. Press Enter to start the session.')
if exist('KbCheck', 'file') ~= 2 || exist('KbName', 'file') ~= 2
    input('Screen is gray and hardware is ready. Press Enter to start the session > ', 's');
    BpodSystem.PluginObjects.V.play(0);
    SoftCodeHandler_BlockSingleInterval(254);
    return
end
KbName('UnifyKeyNames');
enterKey = KbName('Return');
while KbCheck
    pause(0.02)
end
while true
    BpodSystem.PluginObjects.V.play(0);
    [keyDown, ~, keyCode] = KbCheck;
    if keyDown && keyCode(enterKey)
        break
    end
    pause(0.05)
end
while KbCheck
    pause(0.02)
end
BpodSystem.PluginObjects.V.play(0);
SoftCodeHandler_BlockSingleInterval(254);
end

function prepareReadyGrayScreen
global BpodSystem

if ~isfield(BpodSystem, 'PluginObjects') || ~isfield(BpodSystem.PluginObjects, 'V') || isempty(BpodSystem.PluginObjects.V)
    return
end
loadReadyGrayVideo;
try
    SoftCodeHandler_BlockSingleInterval(254);
catch
    try
        BpodSystem.PluginObjects.V.stop;
    catch
    end
end
try
    BpodSystem.PluginObjects.V.play(0);
catch
end
try
    BpodSystem.PluginObjects.V.setSyncPatch(0);
catch
end
end

function holdReadyGrayScreen
global BpodSystem

if ~isfield(BpodSystem, 'PluginObjects') || ~isfield(BpodSystem.PluginObjects, 'V') || isempty(BpodSystem.PluginObjects.V)
    return
end
loadReadyGrayVideo;
try
    BpodSystem.PluginObjects.V.play(0);
catch
end
try
    BpodSystem.PluginObjects.V.setSyncPatch(0);
catch
end
end

function forceGrayScreen
global BpodSystem

if ~isfield(BpodSystem, 'PluginObjects') || ~isfield(BpodSystem.PluginObjects, 'V') || isempty(BpodSystem.PluginObjects.V)
    return
end
loadReadyGrayVideo;
try
    SoftCodeHandler_BlockSingleInterval(254);
catch
end
try
    BpodSystem.PluginObjects.V.play(0);
catch
end
try
    BpodSystem.PluginObjects.V.setSyncPatch(0);
catch
end
end

function loadReadyGrayVideo
global BpodSystem

if ~isfield(BpodSystem, 'PluginObjects') || ~isfield(BpodSystem.PluginObjects, 'V') || isempty(BpodSystem.PluginObjects.V)
    return
end
try
    if isprop(BpodSystem.PluginObjects.V, 'Videos') && numel(BpodSystem.PluginObjects.V.Videos) >= 1 && ~isempty(BpodSystem.PluginObjects.V.Videos{1})
        return
    end
catch
end
width = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
height = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
grayVideo = uint8(127 * ones(height, width, 2));
BpodSystem.PluginObjects.V.loadVideo(1, grayVideo);
BpodSystem.PluginObjects.V.Videos{1}.nFrames = 1;
end

function removeProtocolSubfoldersFromPath
protocolPath = fileparts(mfilename('fullpath'));
entries = dir(protocolPath);
for i = 1:numel(entries)
    if ~entries(i).isdir || startsWith(entries(i).name, '.')
        continue
    end
    folderPath = fullfile(protocolPath, entries(i).name);
    while contains(path, folderPath)
        try
            rmpath(folderPath);
        catch
            break
        end
    end
end
end

function positionParameterGUI
global BpodSystem

if ~isfield(BpodSystem, 'ProtocolFigures') || ~isfield(BpodSystem.ProtocolFigures, 'ParameterGUI')
    return
end
fig = BpodSystem.ProtocolFigures.ParameterGUI;
if ~isgraphics(fig)
    return
end
screenSize = get(0, 'ScreenSize');
position = get(fig, 'Position');
margin = 24;
position(1) = screenSize(1) + margin;
position(2) = screenSize(2) + screenSize(4) - position(4) - margin;
set(fig, 'Position', position);
end

function validateSettings(S)
if S.GUI.MaxTrials < 1
    error('MaxTrials must be at least 1.')
end
if S.GUI.GratingDuration_s <= 0
    error('GratingDuration_s must be positive.')
end
if S.GUI.ShortISIMode == 2 && S.GUI.ShortISIMin_s > S.GUI.ShortISIMax_s
    error('ShortISIMin_s cannot exceed ShortISIMax_s.')
end
if S.GUI.LongISIMode == 2 && S.GUI.LongISIMin_s > S.GUI.LongISIMax_s
    error('LongISIMin_s cannot exceed LongISIMax_s.')
end
if S.GUI.ITIMode == 2 && S.GUI.ITIMin_s > S.GUI.ITIMax_s
    error('ITIMin_s cannot exceed ITIMax_s.')
end
if S.GUI.ITIMode == 2 && S.GUI.ITIMean_s <= 0
    error('ITIMean_s must be positive for exponential ITI.')
end
if S.GUI.PunishITIMode == 2 && S.GUI.PunishITIMin_s > S.GUI.PunishITIMax_s
    error('PunishITIMin_s cannot exceed PunishITIMax_s.')
end
if S.GUI.PunishITIMode == 2 && S.GUI.PunishITIMean_s <= 0
    error('PunishITIMean_s must be positive for exponential punish ITI.')
end
if S.GUI.BlockLength < 1 || S.GUI.BlockMargin < 0 || S.GUI.BlockEdgeTrials < 0 || S.GUI.WarmupBlockNum < 0
    error('BlockLength must be positive; BlockMargin, BlockEdgeTrials, and WarmupBlockNum cannot be negative.')
end
if S.GUI.MostFraction < 0.5 || S.GUI.MostFraction > 1
    error('MostFraction must be between 0.5 and 1.')
end
if S.GUI.OptoMode < 1 || S.GUI.OptoMode > 4
    error('OptoMode must be between 1 and 4.')
end
if S.GUI.OptoFraction < 0 || S.GUI.OptoFraction > 1 || S.GUI.OptoZeroEdgeTrials < 0 || S.GUI.OptoEarlyTrials < 0
    error('OptoFraction must be between 0 and 1; OptoZeroEdgeTrials and OptoEarlyTrials cannot be negative.')
end
if S.GUI.OptoPulseTotalDuration_s < 0 || S.GUI.OptoPulseFrequency_Hz < 0 || S.GUI.OptoPulseDutyCycle_percent < 0 || S.GUI.OptoPulseDutyCycle_percent > 100
    error('Opto pulse total duration and frequency cannot be negative; duty cycle must be between 0 and 100 percent.')
end
if S.GUI.ProbeFraction < 0 || S.GUI.ProbeFraction > 1 || S.GUI.ProbeZeroEdgeTrials < 0
    error('ProbeFraction must be between 0 and 1; ProbeZeroEdgeTrials cannot be negative.')
end
if any([S.GUI.ShortISIFixed_s S.GUI.ShortISIMin_s S.GUI.ShortISIMax_s S.GUI.LongISIFixed_s S.GUI.LongISIMin_s S.GUI.LongISIMax_s S.GUI.ManualITI_s S.GUI.ITIMin_s S.GUI.ITIMax_s S.GUI.ITIMean_s S.GUI.ManualPunishITI_s S.GUI.PunishITIMin_s S.GUI.PunishITIMax_s S.GUI.PunishITIMean_s] < 0)
    error('ISI and ITI values cannot be negative.')
end
if any([S.GUI.PreStimDelay_s S.GUI.SpoutInDelay_s S.GUI.ChoiceWindow_s S.GUI.ChangeMindWindow_s S.GUI.PreOutcomeDelay_s S.GUI.PostRewardDelay_s S.GUI.ServoMoveDelay_s S.GUI.ServoReturnTimeout_s] < 0)
    error('Stimulus, choice, outcome, reward, and servo timing values cannot be negative.')
end
if S.GUI.LeftRewardAmount_uL < 0 || S.GUI.RightRewardAmount_uL < 0
    error('Reward amounts cannot be negative.')
end
if S.GUI.AudioStimVolume < 0 || S.GUI.AudioStimVolume > 1
    error('AudioStimVolume must be between 0 and 1.')
end
if S.GUI.UseSavedImage && ~isfile(fullfile(fileparts(mfilename('fullpath')), 'image.png'))
    error('UseSavedImage is enabled, but image.png was not found in the protocol folder.')
end
end

function confirmDoricOptoSettings(S)
% Ask the user to verify external Doric opto settings before hardware starts.
fprintf('\nDoric opto settings\n');
fprintf('%-24s %s\n', 'OptoMode:', popupValue(S.GUIMeta.OptoMode.String, S.GUI.OptoMode));
fprintf('%-24s %s\n', 'OptoTriggerType:', popupValue(S.GUIMeta.OptoTriggerType.String, S.GUI.OptoTriggerType));
fprintf('%-24s %s\n', 'OptoTriggerMode:', popupValue(S.GUIMeta.OptoTriggerMode.String, S.GUI.OptoTriggerMode));
fprintf('%-24s %.3f s\n', 'Opto total duration:', S.GUI.OptoPulseTotalDuration_s);
fprintf('%-24s %.3f Hz\n', 'Opto frequency:', S.GUI.OptoPulseFrequency_Hz);
fprintf('%-24s %.3f %%\n', 'Opto duty cycle:', S.GUI.OptoPulseDutyCycle_percent);
fprintf('%-24s %s\n', 'EnableOptoStimulus:', onOffText(S.GUI.EnableOptoStimulus));
fprintf('%-24s %s\n', 'EnableOptoSpoutInDelay:', onOffText(S.GUI.EnableOptoSpoutInDelay));
fprintf('%-24s %s\n', 'EnableOptoChoice:', onOffText(S.GUI.EnableOptoChoice));
fprintf('%-24s %s\n', 'EnableOptoPreOutcome:', onOffText(S.GUI.EnableOptoPreOutcome));
fprintf('%-24s %s\n', 'EnableOptoReward:', onOffText(S.GUI.EnableOptoReward));
fprintf('%-24s %s\n', 'EnableOptoPostReward:', onOffText(S.GUI.EnableOptoPostReward));
fprintf('%-24s %s\n', 'EnableOptoPunishITI:', onOffText(S.GUI.EnableOptoPunishITI));
fprintf('%-24s %s\n', 'LED1 control mode:', 'PWM1 gated opto epoch');
fprintf('%-24s %s\n', 'Doric pulse params:', 'check duration/frequency/duty on Doric device');
input('Check these match Doric, then press Enter to continue > ', 's');
end

function value = popupValue(labels, index)
index = round(index);
if index >= 1 && index <= numel(labels)
    value = labels{index};
else
    value = 'Unknown';
end
end

function text = onOffText(value)
if value
    text = 'ON';
else
    text = 'OFF';
end
end

function printTrialInfo(trial, trialType, blockType, probeType, optoType, target, isi, iti, punishITI, stimulus)
trialNames = {'Short', 'Long'};
blockNames = {'50/50', 'Left', 'Right'};
probeNames = {'Off', 'Stim only', 'Servo only'};
modeNames = {'Visual only', 'Audio only', 'Audio + visual'};
fprintf('\nTrial %d\n', trial);
fprintf('%-22s %s\n', 'Trial type:', trialNames{trialType});
fprintf('%-22s %s\n', 'Block type:', blockNames{blockType});
fprintf('%-22s %s\n', 'Probe type:', probeNames{probeType + 1});
fprintf('%-22s %s\n', 'Opto periods:', optoPeriodText(optoType));
fprintf('%-22s %s\n', 'Correct side:', target.Side);
fprintf('%-22s %s\n', 'Stimulus mode:', modeNames{stimulus.Mode});
fprintf('%-22s %s\n', 'Visual source:', sourceName(stimulus.UseSavedImage));
fprintf('%-22s %.4f s\n', 'ISI:', isi);
fprintf('%-22s %.4f s\n', 'Stimulus duration:', stimulus.Duration_s);
fprintf('%-22s %.4f s\n', 'ITI:', iti);
fprintf('%-22s %.4f s\n', 'Punish ITI:', punishITI);
end

function label = sourceName(useSavedImage)
if useSavedImage
    label = 'image.png';
else
    label = 'generated grating';
end
end

function text = optoPeriodText(optoType)
labels = {'stimulus', 'spout-in delay', 'choice', 'pre-outcome', 'reward', 'post-reward', 'punish ITI'};
enabled = find(optoType(:)' ~= 0);
if isempty(enabled)
    text = 'off';
else
    text = strjoin(labels(enabled), ', ');
end
end

function target = trialTarget(S, trialType)
leftIsCorrect = (trialType == 1 && S.GUI.Contingency == 1) || (trialType == 2 && S.GUI.Contingency == 2);
if leftIsCorrect
    target.Side = 'Left';
    target.CorrectLick = 'Port1In';
    target.IncorrectLick = 'Port3In';
    target.Valve = 'Valve1';
    target.ValveTime = max(0, GetValveTimes(S.GUI.LeftRewardAmount_uL, 1));
else
    target.Side = 'Right';
    target.CorrectLick = 'Port3In';
    target.IncorrectLick = 'Port1In';
    target.Valve = 'Valve3';
    target.ValveTime = max(0, GetValveTimes(S.GUI.RightRewardAmount_uL, 3));
end
end

function isi = sampleTrialISI(S, trialType)
if trialType == 1
    isi = sampleValue(S.GUI.ShortISIMode, S.GUI.ShortISIFixed_s, S.GUI.ShortISIMin_s, S.GUI.ShortISIMax_s);
else
    isi = sampleValue(S.GUI.LongISIMode, S.GUI.LongISIFixed_s, S.GUI.LongISIMin_s, S.GUI.LongISIMax_s);
end
end

function iti = sampleTrialITI(S)
iti = sampleITIValue(S.GUI.ITIMode, S.GUI.ManualITI_s, S.GUI.ITIMin_s, S.GUI.ITIMax_s, S.GUI.ITIMean_s);
end

function punishITI = sampleTrialPunishITI(S)
punishITI = sampleITIValue(S.GUI.PunishITIMode, S.GUI.ManualPunishITI_s, S.GUI.PunishITIMin_s, S.GUI.PunishITIMax_s, S.GUI.PunishITIMean_s);
end

function value = sampleITIValue(mode, manualValue, minimum, maximum, meanValue)
if mode == 1
    value = manualValue;
elseif minimum == maximum
    value = minimum;
else
    upperProbability = exp(-minimum / meanValue);
    lowerProbability = exp(-maximum / meanValue);
    value = -meanValue * log(upperProbability - rand * (upperProbability - lowerProbability));
end
end

function value = sampleValue(mode, fixedValue, minimum, maximum)
if mode == 1
    value = fixedValue;
else
    value = minimum + rand * (maximum - minimum);
end
end

function outcome = trialOutcome(rawEvents, target, probeType)
states = rawStates(rawEvents);
if probeType > 0
    outcome = 0;
elseif eventInState(rawEvents, target.IncorrectLick, 'ChoiceWindow') && eventInState(rawEvents, target.CorrectLick, 'ChangeMindWindow')
    outcome = 4;
elseif stateVisited(states, 'Reward') || stateVisited(states, 'NaiveReward') || stateVisited(states, 'PreOutcomeDelay') || stateVisited(states, 'PostRewardDelay')
    outcome = 1;
elseif eventInState(rawEvents, target.IncorrectLick, 'ChoiceWindow') || eventInState(rawEvents, target.IncorrectLick, 'ChangeMindWindow') || stateVisited(states, 'ChangeMindWindow')
    outcome = 2;
else
    outcome = 3;
end
end

function states = rawStates(rawEvents)
states = struct;
if isstruct(rawEvents) && isfield(rawEvents, 'States')
    states = rawEvents.States;
elseif isstruct(rawEvents) && isfield(rawEvents, 'RawEvents') && isfield(rawEvents.RawEvents, 'States')
    states = rawEvents.RawEvents.States;
end
end

function events = rawEventData(rawEvents)
events = struct;
if isstruct(rawEvents) && isfield(rawEvents, 'Events')
    events = rawEvents.Events;
elseif isstruct(rawEvents) && isfield(rawEvents, 'RawEvents') && isfield(rawEvents.RawEvents, 'Events')
    events = rawEvents.RawEvents.Events;
end
end

function visited = stateVisited(states, name)
visited = false;
if ~isfield(states, name)
    return
end
times = states.(name);
visited = ~isempty(times) && any(isfinite(times(:)));
end

function inside = eventInState(rawEvents, eventName, stateName)
inside = false;
events = rawEventData(rawEvents);
states = rawStates(rawEvents);
if ~isfield(events, eventName) || ~isfield(states, stateName)
    return
end
stateTimes = states.(stateName);
eventTimes = events.(eventName);
if isempty(stateTimes) || isempty(eventTimes)
    return
end
if isvector(stateTimes)
    stateTimes = reshape(stateTimes, 1, []);
end
for row = 1:size(stateTimes, 1)
    if size(stateTimes, 2) >= 2 && all(isfinite(stateTimes(row, 1:2)))
        inside = any(eventTimes >= stateTimes(row, 1) & eventTimes <= stateTimes(row, 2));
        if inside
            return
        end
    end
end
end

function moveSpoutsOut(S)
global M
if isempty(M)
    return
end
M.setMotor(0, maestroPosition(S.GUI.RightServoInPos), S.GUI.ServoVelocity);
M.setMotor(1, maestroPosition(S.GUI.LeftServoInPos), S.GUI.ServoVelocity);
end

function port = findMaestroPort
ports = findRegistryPorts('VID_1FFB&PID_0089&MI_00');
for i = 1:numel(ports)
    releaseSerialPort(ports{i});
end
pause(0.2);
availablePorts = reshape(cellstr(serialportlist('available')), 1, []);
ports = ports(ismember(ports, availablePorts));
if numel(ports) ~= 1
    error('Expected one available Pololu Maestro command port, found: %s', strjoin(ports, ', '))
end
port = ports{1};
end

function ports = findRegistryPorts(deviceKey)
command = ['reg query "HKLM\SYSTEM\CurrentControlSet\Enum\USB\' deviceKey '" /s'];
[status, output] = system(command);
if status ~= 0
    ports = {};
    return
end
tokens = regexp(output, 'PortName\s+REG_SZ\s+(COM\d+)', 'tokens');
ports = unique(cellfun(@(token) token{1}, tokens, 'UniformOutput', false), 'stable');
end

function position = maestroPosition(value)
position = value * 0.002 - 3;
end

function printSessionSummary(completedTrials, S)
global BpodSystem

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('block_single_interval_discrimination_202606 complete\n');
fprintf('%-28s %s\n', 'Date:', char(datetime('today', 'Format', 'yyyyMMdd')));
fprintf('%-28s %d / %d\n', 'Trials completed:', completedTrials, round(S.GUI.MaxTrials));
if isfield(BpodSystem, 'Data') && isfield(BpodSystem.Data, 'Outcomes') && ~isempty(BpodSystem.Data.Outcomes)
    outcomes = BpodSystem.Data.Outcomes(1:min(completedTrials, numel(BpodSystem.Data.Outcomes)));
    fprintf('%-28s reward %d, wrong %d, no choice %d, change mind %d\n', ...
        'Outcomes:', sum(outcomes == 1), sum(outcomes == 2), sum(outcomes == 3), sum(outcomes == 4));
end
fprintf('%-28s %s\n', 'Training mode:', popupValue(S.GUIMeta.TrainingMode.String, S.GUI.TrainingMode));
fprintf('%-28s %s\n', 'Contingency:', popupValue(S.GUIMeta.Contingency.String, S.GUI.Contingency));
fprintf('%-28s %s\n', 'Block mode:', popupValue(S.GUIMeta.BlockNum.String, S.GUI.BlockNum));
fprintf('%-28s %d\n', 'Extra 50/50 warmups:', round(S.GUI.WarmupBlockNum));
fprintf('%-28s %d +/- %d trials\n', 'Block length/margin:', round(S.GUI.BlockLength), round(S.GUI.BlockMargin));
fprintf('%-28s %d edge, %.3f majority\n', 'Block edge/most:', round(S.GUI.BlockEdgeTrials), S.GUI.MostFraction);
fprintf('%-28s %s\n', 'Stimulus mode:', popupValue(S.GUIMeta.StimulusMode.String, S.GUI.StimulusMode));
fprintf('%-28s %s\n', 'Visual source:', sourceName(S.GUI.UseSavedImage));
fprintf('%-28s %.3f s, %.3f s\n', 'Pre/stimulus pulse:', S.GUI.PreStimDelay_s, S.GUI.GratingDuration_s);
fprintf('%-28s %.3f Hz, %.3f vol\n', 'Audio freq/volume:', S.GUI.AudioStimFreq_Hz, S.GUI.AudioStimVolume);
fprintf('%-28s %.0f Hz, %.3f dB\n', 'Audio sample/atten:', S.GUI.AudioSamplingRate_Hz, S.GUI.AudioAttenuation_dB);
fprintf('%-28s %.3f ms\n', 'Audio ramp:', S.GUI.AudioRamp_ms);
fprintf('%-28s %s, fixed %.3f, range %.3f-%.3f s\n', 'Short ISI:', popupValue(S.GUIMeta.ShortISIMode.String, S.GUI.ShortISIMode), S.GUI.ShortISIFixed_s, S.GUI.ShortISIMin_s, S.GUI.ShortISIMax_s);
fprintf('%-28s %s, fixed %.3f, range %.3f-%.3f s\n', 'Long ISI:', popupValue(S.GUIMeta.LongISIMode.String, S.GUI.LongISIMode), S.GUI.LongISIFixed_s, S.GUI.LongISIMin_s, S.GUI.LongISIMax_s);
fprintf('%-28s %s\n', 'Opto mode:', popupValue(S.GUIMeta.OptoMode.String, S.GUI.OptoMode));
fprintf('%-28s %.3f, edge %d, early %d\n', 'Opto fraction/edge/early:', S.GUI.OptoFraction, round(S.GUI.OptoZeroEdgeTrials), round(S.GUI.OptoEarlyTrials));
fprintf('%-28s type %s, mode %s\n', 'Opto trigger:', popupValue(S.GUIMeta.OptoTriggerType.String, S.GUI.OptoTriggerType), popupValue(S.GUIMeta.OptoTriggerMode.String, S.GUI.OptoTriggerMode));
fprintf('%-28s %.3f s, %.3f Hz, %.3f %%\n', 'Opto pulse:', S.GUI.OptoPulseTotalDuration_s, S.GUI.OptoPulseFrequency_Hz, S.GUI.OptoPulseDutyCycle_percent);
fprintf('%-28s stim %s, spout delay %s, choice %s, pre outcome %s, reward %s, post %s, punish %s\n', 'Opto periods:', onOffText(S.GUI.EnableOptoStimulus), onOffText(S.GUI.EnableOptoSpoutInDelay), onOffText(S.GUI.EnableOptoChoice), onOffText(S.GUI.EnableOptoPreOutcome), onOffText(S.GUI.EnableOptoReward), onOffText(S.GUI.EnableOptoPostReward), onOffText(S.GUI.EnableOptoPunishITI));
fprintf('%-28s %s\n', 'Chemo:', onOffText(S.GUI.ChemoMode));
fprintf('%-28s %s, %.3f, edge %d\n', 'Probe:', onOffText(S.GUI.ProbeMode), S.GUI.ProbeFraction, round(S.GUI.ProbeZeroEdgeTrials));
fprintf('%-28s %.3f s, %.3f s\n', 'Spout delay/choice:', S.GUI.SpoutInDelay_s, S.GUI.ChoiceWindow_s);
fprintf('%-28s %s, %.3f s\n', 'Change mind:', onOffText(S.GUI.AllowChangeMind), S.GUI.ChangeMindWindow_s);
fprintf('%-28s %.3f s, %.3f s\n', 'Pre outcome/post reward:', S.GUI.PreOutcomeDelay_s, S.GUI.PostRewardDelay_s);
fprintf('%-28s %.3f uL, %.3f uL\n', 'Left/right reward:', S.GUI.LeftRewardAmount_uL, S.GUI.RightRewardAmount_uL);
fprintf('%-28s %s, manual %.3f, range %.3f-%.3f, mean %.3f s\n', 'ITI:', popupValue(S.GUIMeta.ITIMode.String, S.GUI.ITIMode), S.GUI.ManualITI_s, S.GUI.ITIMin_s, S.GUI.ITIMax_s, S.GUI.ITIMean_s);
fprintf('%-28s %s, manual %.3f, range %.3f-%.3f, mean %.3f s\n', 'Punish ITI:', popupValue(S.GUIMeta.PunishITIMode.String, S.GUI.PunishITIMode), S.GUI.ManualPunishITI_s, S.GUI.PunishITIMin_s, S.GUI.PunishITIMax_s, S.GUI.PunishITIMean_s);
fprintf('%s\n\n', repmat('=', 1, 70));
end

function cleanupProtocol
global BpodSystem
global S
global M

try
    if isfield(BpodSystem.PluginObjects, 'H') && ~isempty(BpodSystem.PluginObjects.H)
        BpodSystem.PluginObjects.H.stop;
        delete(BpodSystem.PluginObjects.H);
        BpodSystem.PluginObjects.H = [];
    end
catch
end
try
    if ~isempty(M)
        if ~isempty(S) && isfield(S, 'GUI')
            moveSpoutsOut(S);
        end
        delete(M);
        M = [];
    end
catch
end
try
    if isfield(BpodSystem.PluginObjects, 'V') && ~isempty(BpodSystem.PluginObjects.V)
        BpodSystem.PluginObjects.V.stop;
        BpodSystem.PluginObjects.V.setSyncPatch(0);
        delete(BpodSystem.PluginObjects.V);
        BpodSystem.PluginObjects.V = [];
    end
catch
end
closeSessionFigures;
end

function closeSessionFigures
global BpodSystem

figureNames = {'Session', 'OutcomeLegend', 'ParameterGUI'};
for i = 1:numel(figureNames)
    name = figureNames{i};
    try
        if isfield(BpodSystem, 'ProtocolFigures') && isfield(BpodSystem.ProtocolFigures, name) && isgraphics(BpodSystem.ProtocolFigures.(name))
            close(BpodSystem.ProtocolFigures.(name));
        end
    catch
    end
end
try
    if isfield(BpodSystem, 'ProtocolFigures') && isfield(BpodSystem.ProtocolFigures, 'Session')
        BpodSystem.ProtocolFigures = rmfield(BpodSystem.ProtocolFigures, 'Session');
    end
catch
end
try
    if isfield(BpodSystem, 'ProtocolFigures') && isfield(BpodSystem.ProtocolFigures, 'OutcomeLegend')
        BpodSystem.ProtocolFigures = rmfield(BpodSystem.ProtocolFigures, 'OutcomeLegend');
    end
catch
end
try
    if isfield(BpodSystem, 'ProtocolFigures') && isfield(BpodSystem.ProtocolFigures, 'ParameterGUI')
        BpodSystem.ProtocolFigures = rmfield(BpodSystem.ProtocolFigures, 'ParameterGUI');
    end
catch
end
end
