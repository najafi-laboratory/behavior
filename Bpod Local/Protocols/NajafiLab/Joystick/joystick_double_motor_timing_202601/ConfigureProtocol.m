function S = ConfigureProtocol(BpodSystem)
% Prepare GUI defaults, metadata, and panels.
S = BpodSystem.ProtocolSettings;

if isempty(S) || ~isstruct(S)
    S = struct;
end
if ~isfield(S, 'GUI') || ~isstruct(S.GUI)
    S.GUI = struct;
end
if ~isfield(S, 'GUIMeta') || ~isstruct(S.GUIMeta)
    S.GUIMeta = struct;
end
if ~isfield(S, 'GUIPanels') || ~isstruct(S.GUIPanels)
    S.GUIPanels = struct;
end
if isfield(S, 'ConfigVersion')
    previousVersion = S.ConfigVersion;
else
    previousVersion = 0;
end

% Keep parameter groups compact so GUI panels match task structure.
session = {'MaxTrials', 1000; 'PressMode', 2; 'TrialMode', 4; 'BlockLength', 30; 'BlockLengthEdge', 5; 'ProbeMode', 0; 'ProbeFraction', 0.2; 'ProbeZeroEdgeTrials', 5};
stimulus = {'TimingMode', 1; 'VisualCueDuration_s', 0.1; 'UseGeneratedGrating', 1};
timing = {'ShortDelay_s', 0.5; 'LongDelay_s', 1; 'Press1Window_s', 2; 'ShortPress2Window_s', 3; 'LongPress2Window_s', 3};
joystick = {'PressThreshold', 0.7; 'RetractThreshold', 0.3; 'ServoInPos', 1638; 'ServoOutPos', 50; 'ServoMoveDelay_s', 0.05; 'ServoReturnTimeout_s', 1; 'AssistMode', 1; 'AssistFraction', 0.3};
reward = {'RewardWindowLeft_s', 0.2; 'RewardMaximumWindow_s', 0.5; 'RewardWindowRight_s', 1.5; 'PreRewardDelay_s', 0.5; 'PostRewardDelay_s', 1; 'RewardMode', 1; 'RewardAmount_uL', 3; 'ShortRewardAmount_uL', 3; 'LongRewardAmount_uL', 3};
iti = {'ITIMode', 2; 'ManualITI_s', 1; 'ITIMin_s', 3; 'ITIMax_s', 5; 'ITIMean_s', 4; 'PunishITIMode', 2; 'ManualPunishITI_s', 0; 'PunishITIMin_s', 3; 'PunishITIMax_s', 7; 'PunishITIMean_s', 5};
manipulation = {'OptoMode', 0; 'OptoFraction', 0.35; 'OptoZeroEdgeTrials', 5; 'EnableOptoVisualCue1', 1; 'EnableOptoDelay', 1; 'EnableOptoPreRewardDelay', 1; 'EnableOptoPostReward', 1; 'OptoFrequency_Hz', 50; 'OptoPulseOn_ms', 10; 'ChemoMode', 0};

% Migrate older saved settings into the current field names.
if isfield(S.GUI, 'RewardBefore_s') && ~isfield(S.GUI, 'RewardWindowLeft_s')
    S.GUI.RewardWindowLeft_s = S.GUI.RewardBefore_s;
end
if isfield(S.GUI, 'RewardAfter_s') && ~isfield(S.GUI, 'RewardWindowRight_s')
    S.GUI.RewardWindowRight_s = S.GUI.RewardAfter_s;
end
if isfield(S.GUI, 'RewardDelay_s') && ~isfield(S.GUI, 'PreRewardDelay_s')
    S.GUI.PreRewardDelay_s = S.GUI.RewardDelay_s;
end
if isfield(S.GUI, 'OptoVisualCue1Period') && ~isfield(S.GUI, 'enableoptovisualcue1')
    S.GUI.enableoptovisualcue1 = S.GUI.OptoVisualCue1Period;
end
if isfield(S.GUI, 'OptoPostPress1Period') && ~isfield(S.GUI, 'enableoptodelay')
    S.GUI.enableoptodelay = S.GUI.OptoPostPress1Period;
end
if isfield(S.GUI, 'OptoRewardDelayPeriod') && ~isfield(S.GUI, 'enableoptopostreward')
    S.GUI.enableoptopostreward = S.GUI.OptoRewardDelayPeriod;
end
if isfield(S.GUI, 'OptoPreRewardDelayPeriod') && ~isfield(S.GUI, 'enableoptoprewarddelay')
    S.GUI.enableoptoprewarddelay = S.GUI.OptoPreRewardDelayPeriod;
end
if isfield(S.GUI, 'enableoptovisualcue1') && ~isfield(S.GUI, 'EnableOptoVisualCue1')
    S.GUI.EnableOptoVisualCue1 = S.GUI.enableoptovisualcue1;
end
if isfield(S.GUI, 'enableoptodelay') && ~isfield(S.GUI, 'EnableOptoDelay')
    S.GUI.EnableOptoDelay = S.GUI.enableoptodelay;
end
if isfield(S.GUI, 'enableoptopostreward') && ~isfield(S.GUI, 'EnableOptoPostReward')
    S.GUI.EnableOptoPostReward = S.GUI.enableoptopostreward;
end
if isfield(S.GUI, 'enableoptoprewarddelay') && ~isfield(S.GUI, 'EnableOptoPreRewardDelay')
    S.GUI.EnableOptoPreRewardDelay = S.GUI.enableoptoprewarddelay;
end

groups = {session, stimulus, timing, joystick, reward, iti, manipulation};
parameterNames = vertcat(session(:, 1), stimulus(:, 1), timing(:, 1), joystick(:, 1), reward(:, 1), iti(:, 1), manipulation(:, 1));

% Fill missing fields and drop stale settings from older versions.
for groupIndex = 1:numel(groups)
    group = groups{groupIndex};
    for parameterIndex = 1:size(group, 1)
        name = group{parameterIndex, 1};
        if ~isfield(S.GUI, name)
            S.GUI.(name) = group{parameterIndex, 2};
        end
    end
end
unusedParameters = setdiff(fieldnames(S.GUI), parameterNames);
if ~isempty(unusedParameters)
    S.GUI = rmfield(S.GUI, unusedParameters);
end

% Reset parameters that changed meaning across config versions.
if previousVersion < 4
    S.GUI.VisualCueDuration_s = 0.1;
    S.GUI.ShortDelay_s = 0.5;
    S.GUI.LongDelay_s = 1;
    S.GUI.Press1Window_s = 2;
    S.GUI.ShortPress2Window_s = 3;
    S.GUI.LongPress2Window_s = 3;
    S.GUI.RewardWindowLeft_s = 0.2;
    S.GUI.RewardWindowRight_s = 1.5;
    S.GUI.PreRewardDelay_s = 0.5;
    S.GUI.TimingMode = 2;
end
if previousVersion < 9
    S.GUI.PostRewardDelay_s = 1;
end
if previousVersion < 12
    S.GUI.RewardMaximumWindow_s = 0.5;
end
if previousVersion < 14
    S.GUI.OptoZeroEdgeTrials = 5;
    S.GUI.ProbeZeroEdgeTrials = 5;
end
if previousVersion < 15
    S.GUI.OptoFrequency_Hz = 50;
    S.GUI.OptoPulseOn_ms = 10;
end
if previousVersion < 18
    S.GUI.EnableOptoVisualCue1 = 1;
    S.GUI.EnableOptoDelay = 1;
    S.GUI.EnableOptoPostReward = 1;
end
if previousVersion < 23
    S.GUI.PreRewardDelay_s = 0.5;
    S.GUI.EnableOptoPreRewardDelay = 1;
end

% Configure GUI widget types and menu labels.
S.GUIMeta = struct;
S.GUIPanels = struct;
S.GUIMeta.PressMode.Style = 'popupmenu';
S.GUIMeta.PressMode.String = {'Single Press', 'Double Press'};
S.GUIMeta.TrialMode.Style = 'popupmenu';
S.GUIMeta.TrialMode.String = {'All Short', 'All Long', 'Blocks Short First', 'Blocks Long First'};
S.GUIMeta.TimingMode.Style = 'popupmenu';
S.GUIMeta.TimingMode.String = {'Visual Guided', 'Self Timed'};
S.GUIMeta.ChemoMode.Style = 'checkbox';
S.GUIMeta.UseGeneratedGrating.Style = 'checkbox';
S.GUIMeta.RewardMode.Style = 'popupmenu';
S.GUIMeta.RewardMode.String = {'Same Reward', 'Different Reward'};
S.GUIMeta.ITIMode.Style = 'popupmenu';
S.GUIMeta.ITIMode.String = {'Manual', 'Exponential'};
S.GUIMeta.PunishITIMode.Style = 'popupmenu';
S.GUIMeta.PunishITIMode.String = {'Manual', 'Exponential'};
S.GUIMeta.OptoMode.Style = 'checkbox';
S.GUIMeta.EnableOptoVisualCue1.Style = 'checkbox';
S.GUIMeta.EnableOptoDelay.Style = 'checkbox';
S.GUIMeta.EnableOptoPreRewardDelay.Style = 'checkbox';
S.GUIMeta.EnableOptoPostReward.Style = 'checkbox';
S.GUIMeta.ProbeMode.Style = 'checkbox';
S.GUIMeta.AssistMode.Style = 'checkbox';

% Panel counts: Session 8, Stimulus 3, Timing 5, Joystick 8,
% Reward 9, ITI 10, Manipulation 10.
% This order gives balanced stock Bpod GUI columns after panel reversal.
S.GUIPanels.Joystick = joystick(:, 1)';
S.GUIPanels.Timing = timing(:, 1)';
S.GUIPanels.Reward = reward(:, 1)';
S.GUIPanels.Stimulus = stimulus(:, 1)';
S.GUIPanels.ITI = iti(:, 1)';
S.GUIPanels.Session = session(:, 1)';
S.GUIPanels.Manipulation = manipulation(:, 1)';

S.ConfigVersion = 23;
end
