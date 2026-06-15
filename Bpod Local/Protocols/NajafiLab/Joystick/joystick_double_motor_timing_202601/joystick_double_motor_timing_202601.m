function joystick_double_motor_timing_202601
global BpodSystem
global S
global M
global ProtocolTrialContext

hardwareCleanup = onCleanup(@cleanupProtocolHardware);

protocolPath = fileparts(mfilename('fullpath'));
addpath(fullfile(protocolPath, 'reference'));

% Let the user edit GUI parameters before hardware/session setup.
S = ConfigureProtocol(BpodSystem);
BpodParameterGUI('init', S);
input('Set parameters in the GUI and press Enter to configure the session > ', 's');
S = BpodParameterGUI('sync', S);

validateSettings(S);

if ~isfield(BpodSystem.Data, 'RigName') || isempty(BpodSystem.Data.RigName)
    BpodSystem.Data.ComputerHostName = getenv('COMPUTERNAME');
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
        otherwise
            error('No rig is configured for computer %s.', BpodSystem.Data.ComputerHostName)
    end
end

if ~isempty(M)
    try
        delete(M);
    catch
    end
    M = [];
    pause(0.2);
end

maestroPort = findMaestroPort;
M = PololuMaestro(maestroPort);
BpodSystem.assertModule('RotaryEncoder', 0);
[BpodSystem.PluginObjects.R, encoderPort] = connectRotaryEncoder(maestroPort);
BpodSystem.ModuleUSB.RotaryEncoder1 = encoderPort;
moduleIndex = find(strcmp(BpodSystem.Modules.Name, 'RotaryEncoder1'), 1);
if ~isempty(moduleIndex)
    BpodSystem.Modules.USBport{moduleIndex} = encoderPort;
end
BpodSystem.Data.HardwarePorts.Maestro = maestroPort;
BpodSystem.Data.HardwarePorts.RotaryEncoder = encoderPort;
BpodSystem.PluginObjects.R.sendThresholdEvents = 'on';
BpodSystem.PluginObjects.R.startUSBStream;
BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_Protocol';

% Prepare the visual display and cue video from image.png.
if isfield(BpodSystem.PluginObjects, 'V')
    BpodSystem.PluginObjects.V = [];
end

monitorID = 1;
if ismember(BpodSystem.Data.RigName, {'ImagingRig', 'JoystickRig3', 'JoystickRig4'})
    monitorID = 2;
end

BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(monitorID, 0, [0 0], [180 180], 0);
BpodSystem.PluginObjects.V.SyncPatchIntensity = 255;
BpodSystem.PluginObjects.V.TimerMode = 2;
fps = BpodSystem.PluginObjects.V.DetectedFrameRate;
width = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
height = BpodSystem.PluginObjects.V.ViewPortDimensions(2);

requestedCueDuration = S.GUI.VisualCueDuration_s;
[video, actualCueDuration] = GenerateVisualCueVideo(fullfile(protocolPath, 'image.png'), width, height, fps, requestedCueDuration);
BpodSystem.PluginObjects.V.loadVideo(1, video);
BpodSystem.PluginObjects.V.loadVideo(2, video);
BpodSystem.PluginObjects.V.Videos{1}.nFrames = 1;
BpodSystem.PluginObjects.V.Videos{2}.nFrames = 1;
S.GUI.VisualCueDuration_s = actualCueDuration;
validateSettings(S);
loadedStimulus = requestedCueDuration;
trialConfiguration = [];
itiConfiguration = [];
optoConfiguration = [];
probeConfiguration = [];
trialTypes = [];
optoTypes = [];
probeTypes = [];
itiValues = [];
punishITIValues = [];
currentTrial = 1;

% Put hardware in a quiet ready state before the second Enter.
SoftCodeHandler_Protocol(9);
pause(1);
BpodSystem.PluginObjects.V.play(0);
SoftCodeHandler_Protocol(3);
pause(0.2);
disp('Screen is gray and hardware is ready. Press Enter to start the session.')
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

clear ProtocolPlot
ProtocolPlot('init', ones(1, max(1, round(S.GUI.MaxTrials))), zeros(1, max(1, round(S.GUI.MaxTrials))), zeros(1, max(1, round(S.GUI.MaxTrials))), 0, S);

while currentTrial <= round(S.GUI.MaxTrials)
    % Sync GUI each trial so parameter edits affect the next trial.
    S = BpodParameterGUI('sync', S);
    validateSettings(S);
    if currentTrial > round(S.GUI.MaxTrials)
        break
    end

    % Regenerate planned trials only when block parameters change.
    newTrialConfiguration = [S.GUI.MaxTrials S.GUI.TrialMode S.GUI.BlockLength S.GUI.BlockLengthEdge];
    if ~isequal(newTrialConfiguration, trialConfiguration)
        generatedTypes = GenerateTrials(S);
        if currentTrial > 1 && ~isempty(trialTypes)
            completedTrials = min([currentTrial - 1 numel(generatedTypes) numel(trialTypes)]);
            generatedTypes(1:completedTrials) = trialTypes(1:completedTrials);
        end
        trialTypes = generatedTypes;
        trialConfiguration = newTrialConfiguration;
        BpodSystem.Data.PlannedTrialTypes = trialTypes;
    end

    % Regenerate ITI sequences only when ITI parameters change.
    newITIConfiguration = [S.GUI.MaxTrials S.GUI.ITIMode S.GUI.ManualITI_s S.GUI.ITIMin_s S.GUI.ITIMax_s S.GUI.ITIMean_s S.GUI.PunishITIMode S.GUI.ManualPunishITI_s S.GUI.PunishITIMin_s S.GUI.PunishITIMax_s S.GUI.PunishITIMean_s];
    if ~isequal(newITIConfiguration, itiConfiguration)
        [~, generatedITI, generatedPunishITI] = GenerateTrials(S);
        if currentTrial > 1 && ~isempty(itiValues)
            completedTrials = min([currentTrial - 1 numel(generatedITI) numel(itiValues)]);
            generatedITI(1:completedTrials) = itiValues(1:completedTrials);
            generatedPunishITI(1:completedTrials) = punishITIValues(1:completedTrials);
        end
        itiValues = generatedITI;
        punishITIValues = generatedPunishITI;
        itiConfiguration = newITIConfiguration;
        BpodSystem.Data.PlannedITI = itiValues;
        BpodSystem.Data.PlannedPunishITI = punishITIValues;
    end

    % Regenerate opto and probe schedules when their GUI controls change.
    newOptoConfiguration = [S.GUI.MaxTrials S.GUI.OptoMode S.GUI.OptoFraction];
    if ~isequal(newOptoConfiguration, optoConfiguration)
        generatedOptoTypes = OptoControl('trials', S);
        if currentTrial > 1 && ~isempty(optoTypes)
            completedTrials = min([currentTrial - 1 numel(generatedOptoTypes) numel(optoTypes)]);
            generatedOptoTypes(1:completedTrials) = optoTypes(1:completedTrials);
        end
        optoTypes = generatedOptoTypes;
        optoConfiguration = newOptoConfiguration;
        BpodSystem.Data.PlannedOptoTrialTypes = optoTypes;
    end

    newProbeConfiguration = [S.GUI.MaxTrials S.GUI.ProbeMode S.GUI.ProbeFraction];
    if ~isequal(newProbeConfiguration, probeConfiguration)
        generatedProbeTypes = ProbeControl('trials', S);
        if currentTrial > 1 && ~isempty(probeTypes)
            completedTrials = min([currentTrial - 1 numel(generatedProbeTypes) numel(probeTypes)]);
            generatedProbeTypes(1:completedTrials) = probeTypes(1:completedTrials);
        end
        probeTypes = generatedProbeTypes;
        probeConfiguration = newProbeConfiguration;
        BpodSystem.Data.PlannedProbeTrialTypes = probeTypes;
    end

    % Rebuild visual cue frames if duration changed in the GUI.
    requestedStimulus = S.GUI.VisualCueDuration_s;
    if ~isequal(requestedStimulus, loadedStimulus)
        [video, actualDuration] = GenerateVisualCueVideo(fullfile(protocolPath, 'image.png'), width, height, fps, S.GUI.VisualCueDuration_s);
        BpodSystem.PluginObjects.V.loadVideo(1, video);
        BpodSystem.PluginObjects.V.loadVideo(2, video);
        BpodSystem.PluginObjects.V.Videos{1}.nFrames = 1;
        BpodSystem.PluginObjects.V.Videos{2}.nFrames = 1;
        actualCueDuration = actualDuration;
        loadedStimulus = requestedStimulus;
    end
    S.GUI.VisualCueDuration_s = actualCueDuration;
    validateSettings(S);

    if trialTypes(currentTrial) == 1
        delay = S.GUI.ShortDelay_s;
    else
        delay = S.GUI.LongDelay_s;
    end

    iti = itiValues(currentTrial);
    punishITI = punishITIValues(currentTrial);
    assistTrial = false;
    if currentTrial > 1 && S.GUI.AssistMode && rand < S.GUI.AssistFraction
        previousStates = BpodSystem.Data.RawEvents.Trial{currentTrial - 1}.States;
        assistTrial = isfield(previousStates, 'Press2Early') && any(isfinite(previousStates.Press2Early(:)));
    end
    if S.GUI.RewardMode == 1
        maximumReward = S.GUI.RewardAmount_uL;
    elseif trialTypes(currentTrial) == 1
        maximumReward = S.GUI.ShortRewardAmount_uL;
    else
        maximumReward = S.GUI.LongRewardAmount_uL;
    end

    % Share per-trial reward context with the soft-code handler.
    ProtocolTrialContext.Delay = delay;
    ProtocolTrialContext.MaximumReward_uL = maximumReward;
    ProtocolTrialContext.RewardWindowLeft_s = S.GUI.RewardWindowLeft_s;
    ProtocolTrialContext.RewardMaximumWindow_s = S.GUI.RewardMaximumWindow_s;
    ProtocolTrialContext.RewardWindowRight_s = S.GUI.RewardWindowRight_s;
    ProtocolTrialContext.Press2Clock = [];
    ProtocolTrialContext.Press2Time_s = NaN;
    ProtocolTrialContext.RewardAmount_uL = 0;
    printTrialInfo(currentTrial, trialTypes(currentTrial), optoTypes(currentTrial), probeTypes(currentTrial), assistTrial, delay, iti, punishITI, S);
    ProtocolPlot('update', trialTypes, optoTypes, probeTypes, currentTrial - 1, S);

    % Configure encoder threshold and run the state machine.
    BpodSystem.PluginObjects.R.stopUSBStream;
    pause(0.05);
    BpodSystem.PluginObjects.R.thresholds = S.GUI.PressThreshold;
    BpodSystem.PluginObjects.R.startUSBStream;
    sma = BuildStateMachine(S, delay, trialTypes(currentTrial), optoTypes(currentTrial), assistTrial, iti, punishITI);
    SendStateMachine(sma);
    rawEvents = RunStateMachine;

    if isempty(fieldnames(rawEvents))
        break
    end

    % Save raw events, outcome values, and trial settings.
    BpodSystem.Data = AddTrialEvents(BpodSystem.Data, rawEvents);
    BpodSystem.Data.TrialSettings(currentTrial) = S;
    BpodSystem.Data.TrialTypes(currentTrial) = trialTypes(currentTrial);
    BpodSystem.Data.OptoTrialTypes(currentTrial) = optoTypes(currentTrial);
    BpodSystem.Data.ProbeTrialTypes(currentTrial) = probeTypes(currentTrial);
    BpodSystem.Data.AssistTrial(currentTrial) = assistTrial;
    press2Time = measuredPress2Time(rawEvents.States);
    if ~isfinite(press2Time)
        press2Time = ProtocolTrialContext.Press2Time_s;
    end
    ProtocolTrialContext.Press2Time_s = press2Time;
    BpodSystem.Data.Press2Time(currentTrial) = press2Time;
    BpodSystem.Data.RewardAmount(currentTrial) = ProtocolTrialContext.RewardAmount_uL;
    BpodSystem.Data.ITI(currentTrial) = iti;
    BpodSystem.Data.PunishITI(currentTrial) = punishITI;
    outcome = trialOutcome(rawEvents.States, assistTrial, press2Time, delay, S);
    printTrialResult(outcome, press2Time, ProtocolTrialContext.RewardAmount_uL, maximumReward);

    % Normalize encoder trace so each plotted trial starts at zero.
    encoderData = BpodSystem.PluginObjects.R.readUSBStream();
    if isfield(encoderData, 'EventTimestamps') && ~isempty(encoderData.EventTimestamps)
        encoderStart = encoderData.EventTimestamps(1);
        encoderData.Times = encoderData.Times - encoderStart;
        encoderData.EventTimestamps = encoderData.EventTimestamps - encoderStart;
    elseif ~isempty(encoderData.Times)
        encoderData.Times = encoderData.Times - encoderData.Times(1);
    end
    encoderData.Times = reshape(encoderData.Times, 1, []);
    encoderData.Positions = reshape(encoderData.Positions, 1, []);
    trialDuration = BpodSystem.Data.TrialEndTimestamp(currentTrial) - BpodSystem.Data.TrialStartTimestamp(currentTrial);
    if isempty(encoderData.Times)
        encoderData.Times = [0 trialDuration];
        encoderData.Positions = [0 0];
    else
        valid = encoderData.Times >= 0 & encoderData.Times <= trialDuration;
        encoderData.Times = encoderData.Times(valid);
        encoderData.Positions = encoderData.Positions(valid);
        if isempty(encoderData.Times)
            encoderData.Times = [0 trialDuration];
            encoderData.Positions = [0 0];
        else
            encoderData.Positions = encoderData.Positions - encoderData.Positions(1);
            encoderData.Times = [0 encoderData.Times trialDuration];
            encoderData.Positions = [0 encoderData.Positions encoderData.Positions(end)];
        end
    end
    encoderData.nPositions = numel(encoderData.Positions);
    BpodSystem.Data.EncoderData{currentTrial} = encoderData;
    SaveBpodSessionData;
    ProtocolPlot('update', trialTypes, optoTypes, probeTypes, currentTrial, S);

    if BpodSystem.Status.BeingUsed == 0
        break
    end

    HandlePauseCondition;
    if BpodSystem.Status.BeingUsed == 0
        break
    end

    currentTrial = currentTrial + 1;
end

SoftCodeHandler_Protocol(9);
BpodSystem.PluginObjects.V.stop;
end

function [R, encoderPort] = connectRotaryEncoder(maestroPort)
global BpodSystem

R = [];
cleanupRotaryEncoder;
pause(0.25);

if isfield(BpodSystem.ModuleUSB, 'RotaryEncoder1')
    releaseSerialPort(BpodSystem.ModuleUSB.RotaryEncoder1);
    pause(0.2);
end

availablePorts = reshape(cellstr(serialportlist('available')), 1, []);
excludedPorts = {maestroPort, getPortName(BpodSystem.SerialPort), 'COM1', 'COM3'};
excludedPorts = [excludedPorts findRegistryPorts('VID_1FFB&PID_0089&MI_02')];
if ~isempty(BpodSystem.AnalogSerialPort)
    excludedPorts{end + 1} = getPortName(BpodSystem.AnalogSerialPort);
end

moduleNames = fieldnames(BpodSystem.ModuleUSB);
for i = 1:numel(moduleNames)
    if ~strcmp(moduleNames{i}, 'RotaryEncoder1')
        excludedPorts{end + 1} = BpodSystem.ModuleUSB.(moduleNames{i});
    end
end

candidates = {};
if isfield(BpodSystem.ModuleUSB, 'RotaryEncoder1')
    candidates{end + 1} = BpodSystem.ModuleUSB.RotaryEncoder1;
end
candidates = [candidates findRegistryPorts('VID_16C0&PID_0483&MI_00') availablePorts];
candidates = unique(candidates, 'stable');
candidates = candidates(ismember(candidates, availablePorts));
candidates = candidates(~ismember(candidates, excludedPorts));

attemptErrors = cell(1, numel(candidates));
for i = 1:numel(candidates)
    releaseSerialPort(candidates{i});
    pause(0.1);
    try
        R = RotaryEncoderModule(candidates{i});
        encoderPort = candidates{i};
        return
    catch exception
        attemptErrors{i} = sprintf('%s: %s', candidates{i}, exception.message);
        R = [];
        releaseSerialPort(candidates{i});
    end
end

allPorts = reshape(cellstr(serialportlist('all')), 1, []);
busyPorts = setdiff(allPorts, availablePorts);
error('Rotary encoder not found. Available ports: %s. Busy ports: %s. Attempts: %s', ...
    strjoin(availablePorts, ', '), strjoin(busyPorts, ', '), strjoin(attemptErrors(~cellfun('isempty', attemptErrors)), ' | '))
end

function maestroPort = findMaestroPort
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
maestroPort = ports{1};
end

function ports = findRegistryPorts(deviceKey)
command = ['reg query "HKLM\SYSTEM\CurrentControlSet\Enum\USB\' deviceKey '" /s'];
[status, output] = system(command);
if status == 0
    tokens = regexp(output, 'PortName\s+REG_SZ\s+(COM\d+)', 'tokens');
    ports = cellfun(@(token) token{1}, tokens, 'UniformOutput', false);
    ports = unique(ports, 'stable');
else
    ports = {};
end
end

function portName = getPortName(portObject)
portName = '';
if isempty(portObject)
    return
end
if ischar(portObject) || isstring(portObject)
    portName = char(portObject);
    return
end
if ~isobject(portObject)
    return
end
if isprop(portObject, 'PortName')
    portName = portObject.PortName;
elseif isprop(portObject, 'Port') && isprop(portObject.Port, 'PortName')
    portName = portObject.Port.PortName;
end
end

function cleanupProtocolHardware
global M

cleanupRotaryEncoder;
if ~isempty(M)
    try
        delete(M);
    catch
    end
    M = [];
end
end

function cleanupRotaryEncoder
global BpodSystem

if isempty(BpodSystem)
    return
end

try
    pluginObjects = BpodSystem.PluginObjects;
catch
    return
end

if ~isfield(pluginObjects, 'R')
    return
end

R = pluginObjects.R;
try
    if isa(R, 'RotaryEncoderModule') && isvalid(R)
        try
            R.stopUSBStream;
        catch
        end
        try
            R.sendThresholdEvents = 'off';
        catch
        end
        try
            R.Port = [];
        catch
        end
        delete(R);
    end
catch
end
try
    BpodSystem.PluginObjects.R = [];
catch
end
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

function validateSettings(S)
delays = [S.GUI.ShortDelay_s S.GUI.LongDelay_s];
rewardStarts = delays - S.GUI.RewardWindowLeft_s;
rewardEnds = delays + S.GUI.RewardMaximumWindow_s + S.GUI.RewardWindowRight_s;
press2Windows = [S.GUI.ShortPress2Window_s S.GUI.LongPress2Window_s];
if S.GUI.MaxTrials < 1 || S.GUI.VisualCueDuration_s <= 0 || any(delays <= 0) || S.GUI.Press1Window_s <= 0 || any(press2Windows <= 0)
    error('Trial count, cue duration, delays, and press window must be positive.')
end
if S.GUI.RewardWindowLeft_s <= 0 || S.GUI.RewardMaximumWindow_s < 0 || S.GUI.RewardWindowRight_s <= 0 || any(rewardStarts <= 0)
    error('Reward decay windows must be positive, RewardMaximumWindow_s cannot be negative, and RewardWindowLeft_s must be shorter than both delays.')
end
if any(press2Windows < rewardEnds)
    error('Each press 2 window must include its complete reward window.')
end
if S.GUI.RewardDelay_s < 0 || S.GUI.PostRewardDelay_s < 0 || S.GUI.ServoMoveDelay_s < 0
    error('RewardDelay_s, PostRewardDelay_s, and ServoMoveDelay_s cannot be negative.')
end
if ~ismember(S.GUI.TimingMode, [1 2])
    error('TimingMode must be Visual Guided or Self Timed.')
end
if S.GUI.PressThreshold <= S.GUI.RetractThreshold
    error('PressThreshold must be greater than RetractThreshold.')
end
if S.GUI.BlockLength < 1 || S.GUI.BlockLengthEdge < 0 || S.GUI.ServoReturnTimeout_s <= 0
    error('BlockLength and servo timeout must be positive; BlockLengthEdge cannot be negative.')
end
if any([S.GUI.RewardAmount_uL S.GUI.ShortRewardAmount_uL S.GUI.LongRewardAmount_uL] <= 0)
    error('Reward amounts must be positive.')
end
if S.GUI.ManualITI_s < 0 || S.GUI.ManualPunishITI_s < 0
    error('Manual ITI durations cannot be negative.')
end
if S.GUI.OptoFraction < 0 || S.GUI.OptoFraction > 1
    error('OptoFraction must be between 0 and 1.')
end
if S.GUI.ProbeFraction < 0 || S.GUI.ProbeFraction > 1
    error('ProbeFraction must be between 0 and 1.')
end
if S.GUI.AssistFraction < 0 || S.GUI.AssistFraction > 1
    error('AssistFraction must be between 0 and 1.')
end
if ~ismember(S.GUI.ITIMode, [1 2]) || ~ismember(S.GUI.PunishITIMode, [1 2])
    error('ITI modes must be Manual or Exponential.')
end
if any([S.GUI.ITIMin_s S.GUI.ITIMax_s S.GUI.ITIMean_s S.GUI.PunishITIMin_s S.GUI.PunishITIMax_s S.GUI.PunishITIMean_s] <= 0)
    error('Exponential ITI parameters must be positive.')
end
if S.GUI.ITIMin_s > S.GUI.ITIMax_s || S.GUI.PunishITIMin_s > S.GUI.PunishITIMax_s
    error('ITI minimum values cannot exceed maximum values.')
end
end

function printTrialInfo(trial, trialType, optoType, probeType, assistTrial, delay, iti, punishITI, S)
trialNames = {'Short', 'Long'};
timingNames = {'Visual guided', 'Self timed'};
fprintf('\nTrial %d\n', trial);
fprintf('%-22s %s\n', 'Trial type:', trialNames{trialType});
fprintf('%-22s %s\n', 'Timing mode:', timingNames{S.GUI.TimingMode});
fprintf('%-22s %.3f s\n', 'Perfect timing:', delay);
fprintf('%-22s %.3f / %.3f / %.3f s\n', 'Reward L/Max/R:', S.GUI.RewardWindowLeft_s, S.GUI.RewardMaximumWindow_s, S.GUI.RewardWindowRight_s);
fprintf('%-22s %.3f s\n', 'Press 1 window:', S.GUI.Press1Window_s);
if trialType == 1
    press2Window = S.GUI.ShortPress2Window_s;
else
    press2Window = S.GUI.LongPress2Window_s;
end
fprintf('%-22s %.3f s\n', 'Press 2 window:', press2Window);
fprintf('%-22s %d / %d / %d\n', 'Opto / Probe / Assist:', optoType, probeType, assistTrial);
fprintf('%-22s %.3f / %.3f s\n', 'ITI / Punish ITI:', iti, punishITI);
end

function printTrialResult(outcome, press2Time, rewardAmount, maximumReward)
fprintf('%-22s %s\n', 'Outcome:', outcome);
if isfinite(press2Time)
    fprintf('%-22s %.3f s\n', 'Press 2 timing:', press2Time);
else
    fprintf('%-22s %s\n', 'Press 2 timing:', 'N/A');
end
fprintf('%-22s %.3f uL\n', 'Reward delivered:', rewardAmount);
fprintf('%-22s %.1f%%\n', 'Reward percentage:', 100 * rewardAmount / maximumReward);
end

function outcome = trialOutcome(states, assistTrial, press2Time, delay, S)
if stateVisited(states, 'OutcomeReward') || stateVisited(states, 'Reward')
    if assistTrial
        outcome = 'Assist press';
    else
        outcome = 'Reward';
    end
elseif stateVisited(states, 'Press2Early')
    outcome = 'Early press 2';
elseif stateVisited(states, 'Press2Late')
    outcome = 'Late press 2';
elseif stateVisited(states, 'DidNotPress2') || stateVisited(states, 'ServoBackNoPress2')
    outcome = 'No press 2';
elseif stateVisited(states, 'DidNotPress1') || stateVisited(states, 'ServoBackNoPress1')
    outcome = 'No press 1';
elseif isfinite(press2Time)
    difference = press2Time - delay;
    if difference < -S.GUI.RewardWindowLeft_s
        outcome = 'Early press 2';
    elseif difference > S.GUI.RewardMaximumWindow_s + S.GUI.RewardWindowRight_s
        outcome = 'Late press 2';
    elseif assistTrial
        outcome = 'Assist press';
    else
        outcome = 'Reward';
    end
elseif stateVisited(states, 'WaitForPress2') || stateVisited(states, 'AssistHold') || stateVisited(states, 'AssistWaitForPress2')
    outcome = 'No press 2';
elseif stateVisited(states, 'WaitForPress1')
    outcome = 'No press 1';
else
    outcome = 'Other';
end
end

function elapsed = measuredPress2Time(states)
phaseStart = firstFiniteStateStart(states, {'WaitForPress2', 'AssistHold'});
pressStart = firstFiniteStateStart(states, {'Press2'});
if isfinite(phaseStart) && isfinite(pressStart)
    elapsed = pressStart - phaseStart;
else
    elapsed = NaN;
end
end

function time = firstFiniteStateStart(states, names)
time = NaN;
for i = 1:numel(names)
    if isfield(states, names{i})
        values = states.(names{i});
        if ~isempty(values) && isfinite(values(1))
            time = values(1);
            return
        end
    end
end
end

function value = stateVisited(states, name)
value = false;
if isfield(states, name)
    times = states.(name);
    value = ~isempty(times) && any(isfinite(times(:)));
end
end
