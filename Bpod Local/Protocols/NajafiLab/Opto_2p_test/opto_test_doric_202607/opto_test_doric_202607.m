function opto_test_doric_202607
global BpodSystem
global S

cleanupObject = onCleanup(@closeStimulusWindow);
protocolPath = fileparts(mfilename('fullpath'));
maxTrials = 1000;

S = ConfigureProtocol(BpodSystem);
BpodParameterGUI('init', S);
positionParameterGUI;
input('Set parameters in the GUI and press Enter to configure the session > ', 's');
S = BpodParameterGUI('sync', S);
validateSettings(S);
confirmDoricOptoSettings(S);

BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_Protocol';
monitorID = displayMonitorID;
BpodSystem.PluginObjects.V = PsychToolboxVideoPlayer(monitorID, 0, [0 0], [180 180], 0);
BpodSystem.PluginObjects.V.SyncPatchIntensity = 255;
BpodSystem.PluginObjects.V.TimerMode = 2;
fps = BpodSystem.PluginObjects.V.DetectedFrameRate;
width = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
height = BpodSystem.PluginObjects.V.ViewPortDimensions(2);

loadedImageDuration = S.GUI.ImageDuration_s;
actualImageDuration = loadStimulusVideo(protocolPath, width, height, fps, loadedImageDuration);
S.GUI.ImageDuration_s = actualImageDuration;
validateSettings(S);

BpodSystem.PluginObjects.V.play(0);
SoftCodeHandler_Protocol(3);
pause(0.2);
disp('Screen is gray. Press Enter to start the opto test session.')
input('', 's');

itiValues = GenerateTrials(S, maxTrials);
optoTypes = OptoControl('trials', S, maxTrials);
BpodSystem.Data.PlannedITI = itiValues;
BpodSystem.Data.PlannedOptoTrialTypes = optoTypes;

clear ProtocolPlot
ProtocolPlot('init', optoTypes, 0, S);

trial = 1;
trialConfiguration = currentConfiguration(S);
while trial <= maxTrials
    S = BpodParameterGUI('sync', S);
    validateSettings(S);
    if trial > maxTrials
        break
    end

    if S.GUI.ImageDuration_s ~= loadedImageDuration
        loadedImageDuration = S.GUI.ImageDuration_s;
        actualImageDuration = loadStimulusVideo(protocolPath, width, height, fps, loadedImageDuration);
        S.GUI.ImageDuration_s = actualImageDuration;
        validateSettings(S);
    end

    newConfiguration = currentConfiguration(S);
    if ~isequal(newConfiguration, trialConfiguration) || numel(itiValues) ~= maxTrials
        newITI = GenerateTrials(S, maxTrials);
        newOptoTypes = OptoControl('trials', S, maxTrials);
        completedTrials = min(trial - 1, min(numel(itiValues), numel(newITI)));
        if completedTrials > 0
            newITI(1:completedTrials) = itiValues(1:completedTrials);
            newOptoTypes(:, 1:completedTrials) = optoTypes(:, 1:completedTrials);
        end
        itiValues = newITI;
        optoTypes = newOptoTypes;
        trialConfiguration = newConfiguration;
        BpodSystem.Data.PlannedITI = itiValues;
        BpodSystem.Data.PlannedOptoTrialTypes = optoTypes;
    end

    optoType = OptoControl('trial', S);
    optoTypes(:, trial) = optoType;
    BpodSystem.Data.PlannedOptoTrialTypes(:, trial) = optoType;
    BpodSystem.Data.AssignedOptoTrialCount = trial;

    iti = itiValues(trial);
    printTrialInfo(trial, optoType, iti, S);

    sma = BuildStateMachine(S, optoType, iti);
    SendStateMachine(sma);
    rawEvents = RunStateMachine;
    if ~isstruct(rawEvents) || isempty(fieldnames(rawEvents))
        break
    end

    BpodSystem.Data = AddTrialEvents(BpodSystem.Data, rawEvents);
    completedTrial = completedTrialCount(BpodSystem.Data);
    BpodSystem.Data.TrialSettings(completedTrial) = S;
    BpodSystem.Data.OptoTrialTypes(:, completedTrial) = optoType;
    BpodSystem.Data.ITI(completedTrial) = iti;
    BpodSystem.Data.CompletedTrial(completedTrial) = 1;
    SaveBpodSessionData;
    ProtocolPlot('update', optoTypes, completedTrial, S, completedRawTrial(BpodSystem.Data, completedTrial), optoType);
    printTrialResult(completedTrial);

    if BpodSystem.Status.BeingUsed == 0
        break
    end
    HandlePauseCondition;
    if BpodSystem.Status.BeingUsed == 0
        break
    end
    trial = trial + 1;
end

SoftCodeHandler_Protocol(3);
closeStimulusWindow;
printSessionSummary(S, BpodSystem.Data);
clear cleanupObject
end

function count = completedTrialCount(data)
count = 0;
if isfield(data, 'nTrials') && ~isempty(data.nTrials)
    count = data.nTrials;
elseif isfield(data, 'RawEvents') && isfield(data.RawEvents, 'Trial')
    count = numel(data.RawEvents.Trial);
elseif isfield(data, 'CompletedTrial')
    count = numel(data.CompletedTrial);
end
count = max(1, count);
end

function rawTrial = completedRawTrial(data, trial)
rawTrial = [];
if trial < 1 || ~isfield(data, 'RawEvents') || ~isfield(data.RawEvents, 'Trial')
    return
end
trials = data.RawEvents.Trial;
if iscell(trials)
    if numel(trials) >= trial
        rawTrial = trials{trial};
    end
elseif isstruct(trials)
    if numel(trials) >= trial
        rawTrial = trials(trial);
    end
end
end

function configuration = currentConfiguration(S)
configuration = [ ...
    S.GUI.ITIMode ...
    S.GUI.ManualITI_s ...
    S.GUI.ITIMin_s ...
    S.GUI.ITIMax_s ...
    S.GUI.ITIMean_s ...
    S.GUI.OptoMode ...
    S.GUI.OptoFraction ...
    S.GUI.EnableOptoPreStimDelay ...
    S.GUI.EnableOptoInterval ...
    S.GUI.EnableOptoStim ...
    S.GUI.EnableOptoPostStimDelay ...
    S.GUI.LaserTriggerMode ...
    S.GUI.LaserDuration_s];
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

function monitorID = displayMonitorID
hostName = getenv('COMPUTERNAME');
switch hostName
    case {'COS-3A11406', 'COS-3A14773', 'COS-3A14829'}
        monitorID = 2;
    otherwise
        monitorID = 1;
end
end

function actualDuration = loadStimulusVideo(protocolPath, width, height, fps, requestedDuration)
global BpodSystem

[video, actualDuration] = GenerateVisualCueVideo(fullfile(protocolPath, 'image.png'), width, height, fps, requestedDuration);
BpodSystem.PluginObjects.V.loadVideo(1, video);
BpodSystem.PluginObjects.V.loadVideo(2, video);
BpodSystem.PluginObjects.V.Videos{1}.nFrames = 1;
BpodSystem.PluginObjects.V.Videos{2}.nFrames = 1;
end

function validateSettings(S)
if any([S.GUI.PreStimDelay_s S.GUI.ImageDuration_s S.GUI.ImageInterval_s S.GUI.PostStimDelay_s] < 0)
    error('Trial state durations cannot be negative.')
end
if S.GUI.ImageDuration_s <= 0
    error('ImageDuration_s must be positive.')
end
if S.GUI.OptoMode && ~any([S.GUI.EnableOptoPreStimDelay S.GUI.EnableOptoInterval S.GUI.EnableOptoStim S.GUI.EnableOptoPostStimDelay])
    error('At least one opto epoch must be enabled when OptoMode is on.')
end
if S.GUI.EnableOptoPreStimDelay && S.GUI.PreStimDelay_s <= 0
    error('PreStimDelay_s must be positive when EnableOptoPreStimDelay is on.')
end
if S.GUI.EnableOptoInterval && S.GUI.ImageInterval_s <= 0
    error('ImageInterval_s must be positive when EnableOptoInterval is on.')
end
if S.GUI.EnableOptoPostStimDelay && S.GUI.PostStimDelay_s <= 0
    error('PostStimDelay_s must be positive when EnableOptoPostStimDelay is on.')
end
if S.GUI.OptoFraction < 0 || S.GUI.OptoFraction > 1
    error('OptoFraction must be between 0 and 1.')
end
if ~ismember(S.GUI.ITIMode, [1 2])
    error('ITIMode must be Manual or Exponential.')
end
if S.GUI.ManualITI_s < 0
    error('Manual ITI duration cannot be negative.')
end
if any([S.GUI.ITIMin_s S.GUI.ITIMax_s S.GUI.ITIMean_s] <= 0)
    error('Exponential ITI parameters must be positive.')
end
if S.GUI.ITIMin_s > S.GUI.ITIMax_s
    error('ITI minimum value cannot exceed maximum value.')
end
if ~ismember(S.GUI.LaserTriggerMode, [1 2])
    error('LaserTriggerMode must be Full epoch gate or Fixed duration from onset.')
end
if S.GUI.LaserDuration_s <= 0
    error('LaserDuration_s must be positive.')
end
if S.GUI.OptoFrequency_Hz <= 0 || S.GUI.OptoPulseOn_ms <= 0 || S.GUI.OptoPulseOn_ms / 1000 >= 1 / S.GUI.OptoFrequency_Hz
    error('OptoFrequency_Hz must be positive, and OptoPulseOn_ms must be shorter than one Doric cycle.')
end
end

function confirmDoricOptoSettings(S)
fprintf('\nDoric opto settings to verify\n');
fprintf('%-28s %s\n', 'OptoMode:', onOffText(S.GUI.OptoMode));
fprintf('%-28s %.3f\n', 'OptoFraction:', S.GUI.OptoFraction);
fprintf('%-28s %s\n', 'Pre-stim opto:', onOffText(S.GUI.EnableOptoPreStimDelay));
fprintf('%-28s %s\n', 'Interval opto:', onOffText(S.GUI.EnableOptoInterval));
fprintf('%-28s %s\n', 'Image 2 opto:', onOffText(S.GUI.EnableOptoStim));
fprintf('%-28s %s\n', 'Post-stim opto:', onOffText(S.GUI.EnableOptoPostStimDelay));
fprintf('%-28s %s\n', 'Bpod gate mode:', triggerModeName(S.GUI.LaserTriggerMode));
fprintf('%-28s %.3f s\n', 'LaserDuration_s:', S.GUI.LaserDuration_s);
fprintf('%-28s %.3f Hz\n', 'OptoFrequency_Hz:', S.GUI.OptoFrequency_Hz);
fprintf('%-28s %.3f ms\n', 'OptoPulseOn_ms:', S.GUI.OptoPulseOn_ms);
input('Check these match Doric, then press Enter to continue > ', 's');
end

function printTrialInfo(trial, optoType, iti, S)
fprintf('\nTrial %d\n', trial);
fprintf('%-24s %s\n', 'Opto epochs:', OptoControl('summary', optoType));
fprintf('%-24s %.3f / %.3f / %.3f / %.3f s\n', 'Timing pre/img/int/post:', S.GUI.PreStimDelay_s, S.GUI.ImageDuration_s, S.GUI.ImageInterval_s, S.GUI.PostStimDelay_s);
fprintf('%-24s %.3f s\n', 'ITI:', iti);
end

function printTrialResult(trial)
global BpodSystem

fprintf('%-24s trial %d completed\n', 'Outcome:', trial);
states = [];
if isfield(BpodSystem, 'Data') && isfield(BpodSystem.Data, 'RawEvents') && isfield(BpodSystem.Data.RawEvents, 'Trial')
    trials = BpodSystem.Data.RawEvents.Trial;
    if iscell(trials) && numel(trials) >= trial && isfield(trials{trial}, 'States')
        states = trials{trial}.States;
    elseif isstruct(trials) && numel(trials) >= trial && isfield(trials(trial), 'States')
        states = trials(trial).States;
    end
end
if isstruct(states)
    visitedStates = fieldnames(states)';
    if isempty(visitedStates)
        text = 'None';
    else
        text = strjoin(visitedStates, ', ');
    end
else
    text = sprintf('Unavailable (%s)', class(states));
end
fprintf('%-24s %s\n', 'Visited states:', text);
end

function printSessionSummary(S, data)
completedTrials = 0;
if isfield(data, 'nTrials') && ~isempty(data.nTrials)
    completedTrials = data.nTrials;
elseif isfield(data, 'CompletedTrial')
    completedTrials = sum(data.CompletedTrial ~= 0);
end
fprintf('\n%s\n', repmat('=', 1, 58));
fprintf('%s\n', 'Opto test session summary');
fprintf('%s\n', repmat('-', 1, 58));
fprintf('%-28s %d / %d\n', 'Trials completed:', completedTrials, 1000);
fprintf('%-28s %.3f / %.3f / %.3f / %.3f s\n', 'Pre/img/int/post:', S.GUI.PreStimDelay_s, S.GUI.ImageDuration_s, S.GUI.ImageInterval_s, S.GUI.PostStimDelay_s);
fprintf('%-28s %s, fraction %.3f\n', 'Opto:', onOffText(S.GUI.OptoMode), S.GUI.OptoFraction);
fprintf('%-28s %s\n', 'Bpod gate mode:', triggerModeName(S.GUI.LaserTriggerMode));
fprintf('%-28s %.3f Hz / %.3f ms\n', 'Doric freq / on time:', S.GUI.OptoFrequency_Hz, S.GUI.OptoPulseOn_ms);
fprintf('%-28s %.3f / %.3f / %.3f s\n', 'ITI min / mean / max:', S.GUI.ITIMin_s, S.GUI.ITIMean_s, S.GUI.ITIMax_s);
fprintf('%s\n\n', repmat('=', 1, 58));
end

function text = triggerModeName(mode)
names = {'Full epoch gate', 'Fixed duration from onset'};
text = names{mode};
end

function text = onOffText(value)
if value
    text = 'ON';
else
    text = 'OFF';
end
end

function closeStimulusWindow
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
