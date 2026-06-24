function opto = OptoControl(action, S, varargin)
% Self-contained opto timing helpers for stimulus, choice, and reward periods.
switch action
    case 'actions'
        opto = buildActions(S, varargin{:});
    case 'display'
        opto.Labels = {'Off', 'Stimulus', 'Choice', 'Reward'};
        opto.Colors = [0.70 0.70 0.70; 0.49 0.18 0.56; 0.93 0.69 0.13; 0.30 0.65 0.88];
    otherwise
        error('Unknown opto action: %s', action)
end
end

function opto = buildActions(S, optoType, stimulusPeriod_s)
enabled = false(3, 1);
enabled(1:min(3, numel(optoType))) = optoType(1:min(3, numel(optoType))) ~= 0;
if S.GUI.TrainingMode == 1
    enabled(:) = false;
end

timerIDs = optoTimerIDs();
cancelAll = timerCancelMask(timerIDs);
opto.Timers = offSpecs(timerIDs);
opto.StartActions = {'GlobalTimerCancel', cancelAll, 'PWM1', 0};
opto.StimulusStart = {};
opto.ChoiceOff = {'GlobalTimerCancel', timerIDs(2), 'PWM1', 0};
opto.ChoiceStart = {};
opto.RewardStart = {};
opto.AllOff = opto.StartActions;

if enabled(1)
    opto.Timers(1) = gateSpec(timerIDs(1), stimulusPeriod_s);
    opto.StimulusStart = {'GlobalTimerTrig', timerIDs(1)};
end
if enabled(2)
    opto.Timers(2) = gateSpec(timerIDs(2), S.GUI.ChoiceWindow_s);
    opto.ChoiceStart = {'GlobalTimerTrig', timerIDs(2)};
end
if enabled(3)
    opto.Timers(3) = gateSpec(timerIDs(3), S.GUI.PostRewardDelay_s);
    opto.RewardStart = {'GlobalTimerTrig', timerIDs(3)};
end
end

function specs = offSpecs(timerIDs)
specs = repmat(gateSpec(timerIDs(1), 0.001), 1, numel(timerIDs));
for i = 1:numel(timerIDs)
    specs(i) = gateSpec(timerIDs(i), 0.001);
    specs(i).OnLevel = 0;
    specs(i).OffLevel = 0;
end
end

function spec = gateSpec(timerID, duration)
spec = struct( ...
    'TimerID', timerID, ...
    'Duration', max(0.001, duration), ...
    'OnsetDelay', 0, ...
    'Channel', 'PWM1', ...
    'OnLevel', 255, ...
    'OffLevel', 0, ...
    'Loop', 0, ...
    'LoopInterval', 0);
end

function timerIDs = optoTimerIDs
timerIDs = [10 11 12];
end

function mask = timerCancelMask(timerIDs)
maskLength = max(timerIDs);
mask = repmat('0', 1, maskLength);
mask(maskLength - timerIDs + 1) = '1';
end
