function joystick_double_motor_timing_202601
global BpodSystem
global S
global M
global ProtocolTrialContext

hardwareCleanup = onCleanup(@cleanupProtocolHardware);

protocolPath = fileparts(mfilename('fullpath'));

% Let the user edit GUI parameters before hardware/session setup.
S = ConfigureProtocol(BpodSystem);
BpodParameterGUI('init', S);
positionParameterGUI;
input('Set parameters in the GUI and press Enter to configure the session > ', 's');
S = BpodParameterGUI('sync', S);

% Stop early if GUI values are unsafe or opto settings need confirmation.
validateSettings(S);
confirmDoricOptoSettings(S);

% Identify the rig from the computer hostname.
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

% Reopen Maestro cleanly so stale serial handles do not persist.
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
requestedCueSource = S.GUI.UseGeneratedGrating;
actualCueDuration = loadVisualCueVideo(protocolPath, width, height, fps, requestedCueDuration, requestedCueSource);
S.GUI.VisualCueDuration_s = actualCueDuration;
validateSettings(S);
loadedStimulus = requestedCueDuration;
loadedStimulusSource = requestedCueSource;
trialConfiguration = [];
itiConfiguration = [];
optoConfiguration = [];
probeConfiguration = [];
trialTypes = [];
trialTransitions = [];
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
ProtocolPlot('init', ones(1, max(1, round(S.GUI.MaxTrials))), zeros(4, max(1, round(S.GUI.MaxTrials))), zeros(1, max(1, round(S.GUI.MaxTrials))), 0, S);

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
        trialTransitions = blockTransitionVector(trialTypes);
        trialConfiguration = newTrialConfiguration;
        BpodSystem.Data.PlannedTrialTypes = trialTypes;
        BpodSystem.Data.TrialTransitions = trialTransitions;
        BpodSystem.Data.PlannedTrialTransitions = trialTransitions;
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

    % Regenerate probe first so opto can exclude every probe trial.
    newProbeConfiguration = [S.GUI.MaxTrials S.GUI.ProbeMode S.GUI.ProbeFraction S.GUI.ProbeZeroEdgeTrials trialConfiguration];
    if ~isequal(newProbeConfiguration, probeConfiguration)
        generatedProbeTypes = ProbeControl('trials', S, trialTypes);
        if currentTrial > 1 && ~isempty(probeTypes)
            completedTrials = min([currentTrial - 1 numel(generatedProbeTypes) numel(probeTypes)]);
            generatedProbeTypes(1:completedTrials) = probeTypes(1:completedTrials);
        end
        probeTypes = generatedProbeTypes;
        probeConfiguration = newProbeConfiguration;
        BpodSystem.Data.PlannedProbeTrialTypes = probeTypes;
    end

    % Regenerate opto schedules when opto controls or probe exclusions change.
    newOptoConfiguration = [S.GUI.MaxTrials S.GUI.OptoMode S.GUI.OptoFraction S.GUI.OptoZeroEdgeTrials S.GUI.EnableOptoVisualCue1 S.GUI.EnableOptoDelay S.GUI.EnableOptoPreRewardDelay S.GUI.EnableOptoPostReward S.GUI.OptoFrequency_Hz S.GUI.OptoPulseOn_ms trialConfiguration probeConfiguration];
    if ~isequal(newOptoConfiguration, optoConfiguration)
        generatedOptoTypes = OptoControl('trials', S, trialTypes, probeTypes);
        if currentTrial > 1 && ~isempty(optoTypes)
            completedTrials = min([currentTrial - 1 size(generatedOptoTypes, 2) size(optoTypes, 2)]);
            completedRows = min(size(generatedOptoTypes, 1), size(optoTypes, 1));
            generatedOptoTypes(1:completedRows, 1:completedTrials) = optoTypes(1:completedRows, 1:completedTrials);
        end
        optoTypes = generatedOptoTypes;
        optoConfiguration = newOptoConfiguration;
        BpodSystem.Data.PlannedOptoTrialTypes = optoTypes;
        assertNoProbeOptoOverlap(optoTypes, probeTypes, currentTrial);
    end

    % Rebuild visual cue frames if duration changed in the GUI.
    requestedStimulus = S.GUI.VisualCueDuration_s;
    requestedStimulusSource = S.GUI.UseGeneratedGrating;
    if ~isequal(requestedStimulus, loadedStimulus) || ~isequal(requestedStimulusSource, loadedStimulusSource)
        actualCueDuration = loadVisualCueVideo(protocolPath, width, height, fps, requestedStimulus, requestedStimulusSource);
        loadedStimulus = requestedStimulus;
        loadedStimulusSource = requestedStimulusSource;
    end
    S.GUI.VisualCueDuration_s = actualCueDuration;
    validateSettings(S);

    if trialTypes(currentTrial) == 1
        % Pick the perfect press 2 time for this trial.
        delay = S.GUI.ShortDelay_s;
    else
        delay = S.GUI.LongDelay_s;
    end

    iti = itiValues(currentTrial);
    punishITI = punishITIValues(currentTrial);
    trialS = applyProbeTrialSettings(S, probeTypes(currentTrial));

    % Assist only after an early press 2 on the previous trial.
    assistTrial = false;
    if currentTrial > 1 && trialS.GUI.AssistMode && rand < trialS.GUI.AssistFraction
        previousStates = BpodSystem.Data.RawEvents.Trial{currentTrial - 1}.States;
        assistTrial = stateVisited(previousStates, 'EarlyPress2');
    end
    if trialS.GUI.RewardMode == 1
        maximumReward = trialS.GUI.RewardAmount_uL;
    elseif trialTypes(currentTrial) == 1
        maximumReward = trialS.GUI.ShortRewardAmount_uL;
    else
        maximumReward = trialS.GUI.LongRewardAmount_uL;
    end
    if probeTypes(currentTrial) == 1
        % Probe type 1 omits reward but still runs the timing task.
        maximumReward = 0;
    end

    % Share per-trial reward context with the soft-code handler.
    ProtocolTrialContext.Delay = delay;
    ProtocolTrialContext.MaximumReward_uL = maximumReward;
    ProtocolTrialContext.RewardWindowLeft_s = trialS.GUI.RewardWindowLeft_s;
    ProtocolTrialContext.RewardMaximumWindow_s = trialS.GUI.RewardMaximumWindow_s;
    ProtocolTrialContext.RewardWindowRight_s = trialS.GUI.RewardWindowRight_s;
    ProtocolTrialContext.Press2Clock = [];
    ProtocolTrialContext.Press2Time_s = NaN;
    ProtocolTrialContext.RewardAmount_uL = 0;
    optoType = OptoControl('trial', S, trialTypes, currentTrial, probeTypes);
    optoTypes(:, currentTrial) = optoType;
    BpodSystem.Data.PlannedOptoTrialTypes(:, currentTrial) = optoType;
    assertNoProbeOptoOverlap(optoTypes, probeTypes, currentTrial);
    BpodSystem.Data.AssignedOptoTrialCount = currentTrial;
    printTrialInfo(currentTrial, trialTypes(currentTrial), optoType, probeTypes(currentTrial), assistTrial, delay, iti, punishITI, trialS);
    ProtocolPlot('update', trialTypes, optoTypes, probeTypes, currentTrial - 1, S);

    % Configure encoder threshold and run the state machine.
    BpodSystem.PluginObjects.R.stopUSBStream;
    pause(0.05);
    BpodSystem.PluginObjects.R.thresholds = trialS.GUI.PressThreshold;
    BpodSystem.PluginObjects.R.startUSBStream;
    sma = BuildStateMachine(trialS, delay, trialTypes(currentTrial), optoType, assistTrial, iti, punishITI);
    SendStateMachine(sma);
    rawEvents = RunStateMachine;

    % Empty raw events indicate that the user stopped the protocol.
    if isempty(fieldnames(rawEvents))
        break
    end

    % Save raw events, outcome values, and trial settings.
    BpodSystem.Data = AddTrialEvents(BpodSystem.Data, rawEvents);
    BpodSystem.Data.TrialSettings(currentTrial) = trialS;
    BpodSystem.Data.TrialTypes(currentTrial) = trialTypes(currentTrial);
    BpodSystem.Data.TrialTransitions = trialTransitions;
    BpodSystem.Data.TrialTransition(currentTrial) = trialTransitions(currentTrial);
    BpodSystem.Data.OptoTrialTypes(:, currentTrial) = optoType;
    BpodSystem.Data.ProbeTrialTypes(currentTrial) = probeTypes(currentTrial);
    BpodSystem.Data.ProbeRewardOmitted(currentTrial) = probeTypes(currentTrial) == 1;
    BpodSystem.Data.ChemoTrialTypes(currentTrial) = double(S.GUI.ChemoMode);
    BpodSystem.Data.ChemoTrialType(currentTrial) = double(S.GUI.ChemoMode);
    if any(BpodSystem.Data.ChemoTrialTypes(1:currentTrial) == 1)
        BpodSystem.Data.ChemoTrialTypes(1:currentTrial) = 1;
        BpodSystem.Data.ChemoTrialType(1:currentTrial) = 1;
    end
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
    outcome = trialOutcome(rawEvents.States, assistTrial, press2Time, delay, trialS);
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

% Return hardware and display to a quiet state after the session.
SoftCodeHandler_Protocol(9);
closeStimulusWindow;
printSessionSummary(S, BpodSystem.Data);
end

function positionParameterGUI
% Move the parameter GUI to the top-left corner of the screen.
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

function [R, encoderPort] = connectRotaryEncoder(maestroPort)
% Find and open the rotary encoder without taking another module port.
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

function actualDuration = loadVisualCueVideo(protocolPath, width, height, fps, requestedDuration, useGeneratedGrating)
% Load the same visual cue into both Bpod video slots.
global BpodSystem

[video, actualDuration] = GenerateVisualCueVideo(fullfile(protocolPath, 'image.png'), width, height, fps, requestedDuration, useGeneratedGrating);
BpodSystem.PluginObjects.V.loadVideo(1, video);
BpodSystem.PluginObjects.V.loadVideo(2, video);
BpodSystem.PluginObjects.V.Videos{1}.nFrames = 1;
BpodSystem.PluginObjects.V.Videos{2}.nFrames = 1;
end

function maestroPort = findMaestroPort
% Find the Pololu Maestro command port from the USB registry.
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
% Read Windows registry entries for USB serial COM ports.
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
% Extract a COM port name from a serial object or string.
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
% Always retract hardware and close resources when the function exits.
global M
global S

cleanupRotaryEncoder;
closeStimulusWindow;
if ~isempty(M)
    try
        if ~isempty(S) && isfield(S, 'GUI') && isfield(S.GUI, 'ServoInPos')
            M.setMotor(0, S.GUI.ServoInPos * 0.002 - 3, 0.5);
        end
    catch
    end
    try
        delete(M);
    catch
    end
    M = [];
end
end

function cleanupRotaryEncoder
% Stop encoder streaming and release the serial port.
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
% Delete stale MATLAB serialport objects for one COM port.
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

function trialS = applyProbeTrialSettings(S, probeType)
% Probe type 2 flips visual-guided and self-timed mode for that trial.
trialS = S;
if probeType == 2
    trialS.GUI.TimingMode = 3 - S.GUI.TimingMode;
end
end

function assertNoProbeOptoOverlap(optoTypes, probeTypes, firstTrial)
% Probe trials are reserved and cannot also be opto trials.
if isempty(optoTypes) || isempty(probeTypes)
    return
end
firstTrial = max(1, firstTrial);
lastTrial = min(size(optoTypes, 2), numel(probeTypes));
if firstTrial > lastTrial
    return
end
optoTrials = any(optoTypes(:, firstTrial:lastTrial) ~= 0, 1);
probeTrials = probeTypes(firstTrial:lastTrial) ~= 0;
overlap = find(optoTrials & probeTrials, 1);
if ~isempty(overlap)
    error('Probe trials must not be opto trials. Overlap detected at trial %d.', firstTrial + overlap - 1)
end
end

function closeStimulusWindow
% Close the PsychToolbox video object if it exists.
global BpodSystem

try
    if isfield(BpodSystem.PluginObjects, 'V') && ~isempty(BpodSystem.PluginObjects.V)
        try
            BpodSystem.PluginObjects.V.stop;
        catch
        end
        try
            delete(BpodSystem.PluginObjects.V);
        catch
        end
        BpodSystem.PluginObjects.V = [];
    end
catch
end
end

function validateSettings(S)
% Guard against invalid GUI combinations before building trials.
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
if S.GUI.PreRewardDelay_s < 0 || S.GUI.PostRewardDelay_s < 0 || S.GUI.ServoMoveDelay_s < 0
    error('PreRewardDelay_s, PostRewardDelay_s, and ServoMoveDelay_s cannot be negative.')
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
if S.GUI.OptoZeroEdgeTrials < 0 || S.GUI.OptoZeroEdgeTrials ~= round(S.GUI.OptoZeroEdgeTrials)
    error('OptoZeroEdgeTrials must be a nonnegative integer.')
end
if S.GUI.ProbeZeroEdgeTrials < 0 || S.GUI.ProbeZeroEdgeTrials ~= round(S.GUI.ProbeZeroEdgeTrials)
    error('ProbeZeroEdgeTrials must be a nonnegative integer.')
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
if S.GUI.OptoMode && ~any([S.GUI.EnableOptoVisualCue1 S.GUI.EnableOptoDelay S.GUI.EnableOptoPreRewardDelay S.GUI.EnableOptoPostReward])
    error('At least one opto period must be enabled when OptoMode is on.')
end
if S.GUI.OptoMode && S.GUI.AssistMode
    error('AssistMode must be disabled during opto sessions.')
end
if S.GUI.OptoFrequency_Hz <= 0 || S.GUI.OptoPulseOn_ms <= 0 || S.GUI.OptoPulseOn_ms / 1000 >= 1 / S.GUI.OptoFrequency_Hz
    error('OptoFrequency_Hz must be positive, and OptoPulseOn_ms must be positive and shorter than one Doric square-wave cycle.')
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

function transitions = blockTransitionVector(trialTypes)
% Mark short-to-long and long-to-short block transition trials.
transitions = zeros(size(trialTypes));
for trial = 2:numel(trialTypes)
    if trialTypes(trial) ~= trialTypes(trial - 1)
        if trialTypes(trial) == 2
            transitions(trial) = 1;
        elseif trialTypes(trial) == 1
            transitions(trial) = -1;
        end
    end
end
if ~isempty(transitions)
    transitions(1) = 0;
end
end

function printTrialInfo(trial, trialType, optoType, probeType, assistTrial, delay, iti, punishITI, S)
% Print the settings used by the next trial.
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
fprintf('%-22s %s / %d / %d\n', 'Opto / Probe / Assist:', optoSummary(optoType), probeType, assistTrial);
fprintf('%-22s %.3f / %.3f s\n', 'ITI / Punish ITI:', iti, punishITI);
end

function printTrialResult(outcome, press2Time, rewardAmount, maximumReward)
% Print the measured outcome after each completed trial.
fprintf('%-22s %s\n', 'Outcome:', outcome);
if isfinite(press2Time)
    fprintf('%-22s %.3f s\n', 'Press 2 timing:', press2Time);
else
    fprintf('%-22s %s\n', 'Press 2 timing:', 'N/A');
end
fprintf('%-22s %.3f uL\n', 'Reward delivered:', rewardAmount);
if maximumReward > 0
    fprintf('%-22s %.1f%%\n', 'Reward percentage:', 100 * rewardAmount / maximumReward);
else
    fprintf('%-22s %s\n', 'Reward percentage:', '0.0% (probe reward omitted)');
end
end

function printSessionSummary(S, data)
% Print the final session configuration and completion count.
completedTrials = 0;
if isfield(data, 'nTrials') && ~isempty(data.nTrials)
    completedTrials = data.nTrials;
elseif isfield(data, 'TrialTypes')
    completedTrials = numel(data.TrialTypes);
end

timingNames = {'Visual guided', 'Self timed'};
pressModeNames = {'Single press', 'Double press'};
trialModeNames = {'All short', 'All long', 'Blocks short first', 'Blocks long first'};
rewardModeNames = {'Same reward', 'Different reward'};
itiModeNames = {'Manual', 'Exponential'};

fprintf('\n%s\n', repmat('=', 1, 58));
fprintf('%s\n', 'Session parameters');
fprintf('%s\n', repmat('-', 1, 58));
fprintf('%-28s %d / %d\n', 'Total trials completed:', completedTrials, round(S.GUI.MaxTrials));
fprintf('%-28s %s\n', 'Press mode:', pressModeNames{S.GUI.PressMode});
fprintf('%-28s %s\n', 'Trial mode:', trialModeNames{S.GUI.TrialMode});
fprintf('%-28s %s\n', 'Timing mode:', timingNames{S.GUI.TimingMode});
fprintf('%-28s %.3f / %.3f s\n', 'Short / long delay:', S.GUI.ShortDelay_s, S.GUI.LongDelay_s);
fprintf('%-28s %.3f s\n', 'Visual cue duration:', S.GUI.VisualCueDuration_s);
fprintf('%-28s %.3f s\n', 'Press 1 window:', S.GUI.Press1Window_s);
fprintf('%-28s %.3f / %.3f s\n', 'Short / long press 2:', S.GUI.ShortPress2Window_s, S.GUI.LongPress2Window_s);
fprintf('%-28s %.3f / %.3f / %.3f s\n', 'Reward L / Max / R:', S.GUI.RewardWindowLeft_s, S.GUI.RewardMaximumWindow_s, S.GUI.RewardWindowRight_s);
fprintf('%-28s %.3f / %.3f s\n', 'Pre reward / post delay:', S.GUI.PreRewardDelay_s, S.GUI.PostRewardDelay_s);
fprintf('%-28s %s\n', 'Reward mode:', rewardModeNames{S.GUI.RewardMode});
fprintf('%-28s %.3f / %.3f / %.3f uL\n', 'Reward amounts:', S.GUI.RewardAmount_uL, S.GUI.ShortRewardAmount_uL, S.GUI.LongRewardAmount_uL);
fprintf('%-28s %.3f / %.3f\n', 'Press / retract threshold:', S.GUI.PressThreshold, S.GUI.RetractThreshold);
fprintf('%-28s %s, %.3f s\n', 'ITI:', itiModeNames{S.GUI.ITIMode}, S.GUI.ManualITI_s);
fprintf('%-28s %.3f / %.3f / %.3f s\n', 'ITI min / mean / max:', S.GUI.ITIMin_s, S.GUI.ITIMean_s, S.GUI.ITIMax_s);
fprintf('%-28s %s, %.3f s\n', 'Punish ITI:', itiModeNames{S.GUI.PunishITIMode}, S.GUI.ManualPunishITI_s);
fprintf('%-28s %.3f / %.3f / %.3f s\n', 'Punish min / mean / max:', S.GUI.PunishITIMin_s, S.GUI.PunishITIMean_s, S.GUI.PunishITIMax_s);
fprintf('%-28s %s\n', 'Opto enabled:', onOffText(S.GUI.OptoMode));
fprintf('%-28s %s\n', 'Assist disabled for opto:', onOffText(~S.GUI.OptoMode || ~S.GUI.AssistMode));
fprintf('%-28s %.3f, edge %d\n', 'Opto fraction / zero edge:', S.GUI.OptoFraction, round(S.GUI.OptoZeroEdgeTrials));
fprintf('%-28s %s\n', 'EnableOptoVisualCue1:', onOffText(S.GUI.EnableOptoVisualCue1));
fprintf('%-28s %s\n', 'EnableOptoDelay:', onOffText(S.GUI.EnableOptoDelay));
fprintf('%-28s %s\n', 'EnableOptoPreRewardDelay:', onOffText(S.GUI.EnableOptoPreRewardDelay));
fprintf('%-28s %s\n', 'EnableOptoPostReward:', onOffText(S.GUI.EnableOptoPostReward));
fprintf('%-28s %.3f Hz / %.3f ms\n', 'Doric freq / on time:', S.GUI.OptoFrequency_Hz, S.GUI.OptoPulseOn_ms);
fprintf('%-28s %s\n', 'LED1 control mode:', 'Gated opto epoch');
fprintf('%-28s %s\n', 'Probe enabled:', onOffText(S.GUI.ProbeMode));
fprintf('%-28s %.3f, edge %d\n', 'Probe fraction / zero edge:', S.GUI.ProbeFraction, round(S.GUI.ProbeZeroEdgeTrials));
fprintf('%-28s %s, fraction %.3f\n', 'Assist enabled:', onOffText(S.GUI.AssistMode), S.GUI.AssistFraction);
fprintf('%-28s %s\n', 'Chemo enabled:', onOffText(S.GUI.ChemoMode));
fprintf('%s\n\n', repmat('=', 1, 58));
end

function confirmDoricOptoSettings(S)
% Ask the user to verify that Doric square-wave settings match the GUI.
fprintf('\nDoric opto square-wave settings\n');
fprintf('%-24s %s\n', 'Assist disabled for opto:', onOffText(~S.GUI.OptoMode || ~S.GUI.AssistMode));
fprintf('%-24s %s\n', 'EnableOptoVisualCue1:', onOffText(S.GUI.EnableOptoVisualCue1));
fprintf('%-24s %s\n', 'EnableOptoDelay:', onOffText(S.GUI.EnableOptoDelay));
fprintf('%-24s %s\n', 'EnableOptoPreRewardDelay:', onOffText(S.GUI.EnableOptoPreRewardDelay));
fprintf('%-24s %s\n', 'EnableOptoPostReward:', onOffText(S.GUI.EnableOptoPostReward));
fprintf('%-24s %.3f Hz\n', 'OptoFrequency_Hz:', S.GUI.OptoFrequency_Hz);
fprintf('%-24s %.3f ms\n', 'OptoPulseOn_ms:', S.GUI.OptoPulseOn_ms);
fprintf('%-24s %s\n', 'LED1 control mode:', 'Gated opto epoch');
input('Check these match Doric, then press Enter to continue > ', 's');
end

function text = onOffText(value)
% Convert logical settings into terminal text.
if value
    text = 'ON';
else
    text = 'OFF';
end
end

function text = optoSummary(optoType)
% Convert one opto column to compact terminal text.
labels = {'Cue1', 'Delay', 'PreRewardDelay', 'PostReward'};
optoType = optoType(:) ~= 0;
if numel(optoType) == 3
    optoType = [optoType(1:2); false; optoType(3)];
elseif numel(optoType) < numel(labels)
    optoType = [optoType; false(numel(labels) - numel(optoType), 1)];
elseif numel(optoType) > numel(labels)
    optoType = optoType(1:numel(labels));
end
enabled = labels(optoType);
if isempty(enabled)
    text = 'Off';
else
    text = strjoin(enabled, '+');
end
end

function outcome = trialOutcome(states, assistTrial, press2Time, delay, S)
% Convert state visits and press timing into one human-readable outcome.
if stateVisited(states, 'LeverRetractFinal') || stateVisited(states, 'Reward')
    if assistTrial
        outcome = 'Assist press';
    else
        outcome = 'Reward';
    end
elseif stateVisited(states, 'EarlyPress2')
    outcome = 'Early press 2';
elseif stateVisited(states, 'Press2Late')
    outcome = 'Late press 2';
elseif stateVisited(states, 'DidNotPress2')
    outcome = 'No press 2';
elseif stateVisited(states, 'DidNotPress1')
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
elseif stateVisited(states, 'WaitForPress2') || stateVisited(states, 'PrePress2Delay') || stateVisited(states, 'Assist')
    outcome = 'No press 2';
elseif stateVisited(states, 'WaitForPress1')
    outcome = 'No press 1';
else
    outcome = 'Other';
end
end

function elapsed = measuredPress2Time(states)
% Measure press 2 time from press-window onset.
phaseStart = firstFiniteStateStart(states, {'PrePress2Delay', 'WaitForPress2'});
pressStart = firstFiniteStateStart(states, {'Press2'});
if isfinite(phaseStart) && isfinite(pressStart)
    elapsed = pressStart - phaseStart;
else
    elapsed = NaN;
end
end

function time = firstFiniteStateStart(states, names)
% Return the first valid start time from a list of state names.
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
% Test whether a Bpod state was entered at least once.
value = false;
if isfield(states, name)
    times = states.(name);
    value = ~isempty(times) && any(isfinite(times(:)));
end
end
