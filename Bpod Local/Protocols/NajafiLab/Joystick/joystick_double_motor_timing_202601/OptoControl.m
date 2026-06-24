function output = OptoControl(action, S, varargin)
% Generate opto trial tags and plotting metadata.
switch action
    case 'trials'
        % Draw period-wise opto tags only when opto mode is enabled.
        trialTypes = [];
        if ~isempty(varargin)
            trialTypes = varargin{1};
        end

        nTrials = round(S.GUI.MaxTrials);
        output = zeros(3, nTrials);
        if S.GUI.OptoMode
            enabledPeriods = enabledOptoPeriods(S);
            eligible = eligibleTagTrials(S, nTrials, trialTypes);
            nOpto = min(numel(eligible), max(0, round(S.GUI.OptoFraction * numel(eligible))));
            if nOpto > 0 && any(enabledPeriods)
                indices = eligible(randperm(numel(eligible), nOpto));
                output(:, indices) = repmat(enabledPeriods, 1, nOpto);
            end
        end
    case 'trial'
        % Assign one current trial using the latest GUI values.
        trialTypes = varargin{1};
        trial = varargin{2};
        output = zeros(3, 1);
        if S.GUI.OptoMode
            enabledPeriods = enabledOptoPeriods(S);
            if any(enabledPeriods) && eligibleTrial(S, round(S.GUI.MaxTrials), trialTypes, trial) && rand < S.GUI.OptoFraction
                output = enabledPeriods;
            end
        end
    case 'actions'
        % Return state-machine timer specs and trigger actions for this trial.
        optoType = normalizeOptoType(varargin{1});
        if numel(optoType) ~= 3
            error('Opto trial type must be a 3-row vector: cue, delay, post reward.')
        end
        output.Enabled = any(optoType);
        output.TrialType = optoType;
        output.Timers = offSpec(0.001);
        output.StartActions = {'GlobalTimerCancel', optoTimerID, 'PWM1', 0};
        output.VisualStimulus1Actions = {};
        output.VisualStimulus1OffActions = {};
        output.LeverRetract1Actions = {};
        output.RewardLeverRetractActions = {};
        output.PreRewardDelayActions = {};
        output.PostRewardDelayActions = {};
        output.LeverRetractFinalActions = {};

        if optoType(1)
            output.VisualStimulus1Actions = {'PWM1', 255};
            output.VisualStimulus1OffActions = {'PWM1', 0};
        end
        if optoType(2)
            press2Window = varargin{3};
            output.Timers = gateSpec(press2Window);
            output.LeverRetract1Actions = {'GlobalTimerTrig', optoTimerID};
            output.RewardLeverRetractActions = {'GlobalTimerCancel', optoTimerID, 'PWM1', 0};
        end
        if optoType(3)
            output.PostRewardDelayActions = {'PWM1', 255};
            output.LeverRetractFinalActions = {'PWM1', 0};
        end
    case 'waveform'
        % Provide high intervals for plotting the LED 1 waveform.
        duration = varargin{1};
        output = waveformIntervals(S, duration);
    case 'display'
        % Keep labels and colors centralized for plotting.
        output.Labels = {'Off', 'Cue 1', 'Delay', 'Post Reward'};
        output.Colors = [0.85 0.85 0.85; 0.58 0.58 0.58; 0.32 0.32 0.32; 0.08 0.08 0.08];
    otherwise
        error('Unknown opto action: %s', action)
end
end

function timerSpec = gateSpec(duration)
% Keep LED1 high until the timer duration elapses or it is cancelled.
timerSpec = struct( ...
    'TimerID', optoTimerID, ...
    'Duration', duration, ...
    'OnsetDelay', 0, ...
    'Channel', 'PWM1', ...
    'OnLevel', 255, ...
    'OffLevel', 0, ...
    'Loop', 0, ...
    'LoopInterval', 0);
end

function timerSpec = offSpec(onsetDelay)
% Use a silent timer to switch LED1 off after a delay.
timerSpec = struct( ...
    'TimerID', optoTimerID, ...
    'Duration', 0.001, ...
    'OnsetDelay', onsetDelay, ...
    'Channel', 'PWM1', ...
    'OnLevel', 0, ...
    'OffLevel', 0, ...
    'Loop', 0, ...
    'LoopInterval', 0);
end

function timerID = optoTimerID
% Reserve one global timer for opto gating.
timerID = 10;
end

function intervals = waveformIntervals(~, requestedDuration)
% Plotting uses one continuous high interval for each opto epoch.
intervals = [0 requestedDuration];
end

function periods = enabledOptoPeriods(S)
% Convert GUI checkboxes into one opto column: cue, delay, post reward.
periods = [S.GUI.EnableOptoVisualCue1; S.GUI.EnableOptoDelay; S.GUI.EnableOptoPostReward] ~= 0;
end

function optoType = normalizeOptoType(optoType)
% Accept the current 3-row representation and legacy scalar values.
if isempty(optoType)
    optoType = zeros(3, 1);
elseif isscalar(optoType)
    legacy = optoType;
    optoType = zeros(3, 1);
    if ismember(legacy, 1:3)
        optoType(legacy) = 1;
    end
else
    optoType = optoType(:) ~= 0;
end
end

function eligible = eligibleTrial(S, nTrials, trialTypes, trial)
% Test one trial against the same first-block and block-edge rules.
eligibleTrials = eligibleTagTrials(S, nTrials, trialTypes);
eligible = ismember(trial, eligibleTrials);
end

function eligible = eligibleTagTrials(S, nTrials, trialTypes)
% Exclude block edges and the first block from opto tagging.
edge = max(0, round(S.GUI.OptoZeroEdgeTrials));
blocked = false(1, nTrials);

if ~isempty(trialTypes)
    blockStarts = [1 find(diff(trialTypes) ~= 0) + 1];
    blockEnds = [blockStarts(2:end) - 1 nTrials];
    for i = 1:numel(blockStarts)
        firstEdge = blockStarts(i):min(blockEnds(i), blockStarts(i) + edge - 1);
        lastEdge = max(blockStarts(i), blockEnds(i) - edge + 1):blockEnds(i);
        blocked(firstEdge) = true;
        blocked(lastEdge) = true;
    end
    blocked(blockStarts(1):blockEnds(1)) = true;
else
    firstBlockEnd = min(nTrials, round(S.GUI.BlockLength + S.GUI.BlockLengthEdge));
    blocked(1:firstBlockEnd) = true;
end

eligible = find(~blocked);
end
