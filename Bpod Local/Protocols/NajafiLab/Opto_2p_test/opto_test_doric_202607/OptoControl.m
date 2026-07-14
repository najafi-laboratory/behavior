function output = OptoControl(action, S, varargin)
% Generate opto trial tags, Bpod actions, and plotting metadata.
switch action
    case 'trials'
        if isempty(varargin)
            nTrials = 1000;
        else
            nTrials = varargin{1};
        end
        output = zeros(4, nTrials);
        if S.GUI.OptoMode
            enabledEpochs = enabledOptoEpochs(S);
            assigned = rand(1, nTrials) < S.GUI.OptoFraction;
            output(:, assigned) = repmat(enabledEpochs, 1, sum(assigned));
        end
    case 'trial'
        output = zeros(4, 1);
        if S.GUI.OptoMode && rand < S.GUI.OptoFraction
            output = enabledOptoEpochs(S);
        end
    case 'actions'
        output = optoActions(S, normalizeOptoType(varargin{1}));
    case 'display'
        output.Labels = {'Off', 'Pre-stim', 'Interval', 'Image 2', 'Post-stim'};
        output.Colors = [0.85 0.85 0.85; 0.35 0.35 0.35; 0.55 0.55 0.55; 0.05 0.05 0.05; 0.70 0.70 0.70];
    case 'summary'
        labels = {'Pre-stim', 'Interval', 'Image 2', 'Post-stim'};
        optoType = normalizeOptoType(S);
        enabled = labels(optoType ~= 0);
        if isempty(enabled)
            output = 'Off';
        else
            output = strjoin(enabled, '+');
        end
    otherwise
        error('Unknown opto action: %s', action)
end
end

function actions = optoActions(S, optoType)
actions.Timers = timerSpecs(S, optoType);
actions.Start = {'PWM1', 0};
actions.PreStimDelay = {'PWM1', 0};
actions.Image1Display = {'PWM1', 0};
actions.ImageInterval = {'PWM1', 0};
actions.Image2Display = {'PWM1', 0};
actions.PostStimDelay = {'PWM1', 0};
actions.ITI = {'PWM1', 0};

if S.GUI.LaserTriggerMode == 1
    if optoType(1)
        actions.PreStimDelay = {'PWM1', 255};
    end
    if optoType(2)
        actions.ImageInterval = {'PWM1', 255};
    end
    if optoType(3)
        actions.Image2Display = {'PWM1', 255};
    end
    if optoType(4)
        actions.PostStimDelay = {'PWM1', 255};
    end
else
    if optoType(1)
        actions.PreStimDelay = {'GlobalTimerTrig', 1};
    end
    if optoType(2)
        actions.ImageInterval = {'GlobalTimerTrig', 2};
    end
    if optoType(3)
        actions.Image2Display = {'GlobalTimerTrig', 3};
    end
    if optoType(4)
        actions.PostStimDelay = {'GlobalTimerTrig', 4};
    end
end
end

function specs = timerSpecs(S, optoType)
durations = [ ...
    S.GUI.PreStimDelay_s ...
    S.GUI.ImageInterval_s ...
    S.GUI.ImageDuration_s ...
    S.GUI.PostStimDelay_s];
specs = struct('TimerID', {}, 'Duration', {}, 'Channel', {}, 'OnLevel', {}, 'OffLevel', {});
if S.GUI.LaserTriggerMode ~= 2
    return
end
for i = 1:4
    if optoType(i)
        specs(end + 1) = struct( ...
            'TimerID', i, ...
            'Duration', min(S.GUI.LaserDuration_s, durations(i)), ...
            'Channel', 'PWM1', ...
            'OnLevel', 255, ...
            'OffLevel', 0);
    end
end
end

function periods = enabledOptoEpochs(S)
periods = [S.GUI.EnableOptoPreStimDelay; S.GUI.EnableOptoInterval; S.GUI.EnableOptoStim; S.GUI.EnableOptoPostStimDelay] ~= 0;
end

function optoType = normalizeOptoType(optoType)
optoType = optoType(:) ~= 0;
if numel(optoType) < 4
    optoType = [optoType; false(4 - numel(optoType), 1)];
elseif numel(optoType) > 4
    optoType = optoType(1:4);
end
end
