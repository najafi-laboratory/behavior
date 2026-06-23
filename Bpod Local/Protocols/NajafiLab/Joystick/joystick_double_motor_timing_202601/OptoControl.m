function output = OptoControl(action, S, varargin)
% Generate opto trial tags and plotting metadata.
switch action
    case 'trials'
        % Draw opto trial types only when opto mode is enabled.
        trialTypes = [];
        if ~isempty(varargin)
            trialTypes = varargin{1};
        end

        nTrials = round(S.GUI.MaxTrials);
        output = zeros(1, nTrials);
        if S.GUI.OptoMode
            enabledTypes = enabledOptoTypes(S);
            eligible = eligibleTagTrials(S, nTrials, trialTypes);
            nOpto = min(numel(eligible), max(0, round(S.GUI.OptoFraction * numel(eligible))));
            if nOpto > 0 && ~isempty(enabledTypes)
                indices = eligible(randperm(numel(eligible), nOpto));
                output(indices) = enabledTypes(randi(numel(enabledTypes), 1, nOpto));
            end
        end
    case 'actions'
        % Return state-machine timer specs and trigger actions for this opto type.
        optoType = varargin{1};
        if ~ismember(optoType, 0:3)
            error('Opto trial type must be 0, 1, 2, or 3.')
        end
        output.Enabled = optoType > 0;
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

        if optoType == 1
            output.VisualStimulus1Actions = {'PWM1', 255};
            output.VisualStimulus1OffActions = {'PWM1', 0};
        elseif optoType == 2
            press2Window = varargin{3};
            output.Timers = gateSpec(press2Window);
            output.LeverRetract1Actions = {'GlobalTimerTrig', optoTimerID};
            output.RewardLeverRetractActions = {'GlobalTimerCancel', optoTimerID, 'PWM1', 0};
        elseif optoType == 3
            output.PostRewardDelayActions = {'PWM1', 255};
            output.LeverRetractFinalActions = {'PWM1', 0};
        end
    case 'waveform'
        % Provide high intervals for plotting the LED 1 waveform.
        duration = varargin{1};
        output = waveformIntervals(S, duration);
    case 'display'
        % Keep labels and colors centralized for plotting.
        output.Labels = {'Off', 'Visual Cue 1', 'Delay', 'Post Reward'};
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

function types = enabledOptoTypes(S)
% Convert GUI checkboxes into numeric opto trial types.
types = [];
if S.GUI.EnableOptoVisualCue1
    types(end + 1) = 1;
end
if S.GUI.EnableOptoDelay
    types(end + 1) = 2;
end
if S.GUI.EnableOptoPostReward
    types(end + 1) = 3;
end
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
