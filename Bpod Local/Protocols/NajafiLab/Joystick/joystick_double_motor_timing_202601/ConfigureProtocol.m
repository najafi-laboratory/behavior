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
session = {'MaxTrials', 1000; 'PressMode', 2; 'TrialMode', 4; 'BlockLength', 30; 'BlockLengthEdge', 5};
stimulus = {'TimingMode', 1; 'VisualCueDuration_s', 0.1; 'UseGeneratedGrating', 0};
timing = {'ShortDelay_s', 0.5; 'LongDelay_s', 1; 'Press1Window_s', 2; 'ShortPress2Window_s', 3; 'LongPress2Window_s', 3};
joystick = {'PressThreshold', 0.7; 'RetractThreshold', 0.3; 'ServoInPos', 1678; 'ServoOutPos', 50; 'ServoMoveDelay_s', 0.05; 'ServoReturnTimeout_s', 1};
assist = {'AssistMode', 1; 'AssistFraction', 0.3};
reward = {'RewardWindowLeft_s', 0.2; 'RewardMaximumWindow_s', 0.5; 'RewardWindowRight_s', 1.5; 'RewardDelay_s', 0.1; 'PostRewardDelay_s', 1; 'RewardMode', 1; 'RewardAmount_uL', 3; 'ShortRewardAmount_uL', 3; 'LongRewardAmount_uL', 3};
iti = {'ITIMode', 2; 'ManualITI_s', 1; 'ITIMin_s', 3; 'ITIMax_s', 5; 'ITIMean_s', 4; 'PunishITIMode', 2; 'ManualPunishITI_s', 0; 'PunishITIMin_s', 3; 'PunishITIMax_s', 7; 'PunishITIMean_s', 5};
opto = {'OptoMode', 0; 'OptoFraction', 0.35; 'OptoZeroEdgeTrials', 5; 'OptoFrequency_Hz', 20; 'OptoPulseOn_ms', 10};
probe = {'ProbeMode', 0; 'ProbeFraction', 0.2; 'ProbeZeroEdgeTrials', 5};
chemo = {'ChemoMode', 0};

if isfield(S.GUI, 'RewardBefore_s') && ~isfield(S.GUI, 'RewardWindowLeft_s')
    S.GUI.RewardWindowLeft_s = S.GUI.RewardBefore_s;
end
if isfield(S.GUI, 'RewardAfter_s') && ~isfield(S.GUI, 'RewardWindowRight_s')
    S.GUI.RewardWindowRight_s = S.GUI.RewardAfter_s;
end

groups = {session, stimulus, timing, joystick, assist, reward, iti, opto, probe, chemo};
parameterNames = vertcat(session(:, 1), stimulus(:, 1), timing(:, 1), joystick(:, 1), assist(:, 1), reward(:, 1), iti(:, 1), opto(:, 1), probe(:, 1), chemo(:, 1));

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

if previousVersion < 4
    S.GUI.VisualCueDuration_s = 0.1;
    S.GUI.ShortDelay_s = 0.5;
    S.GUI.LongDelay_s = 1;
    S.GUI.Press1Window_s = 2;
    S.GUI.ShortPress2Window_s = 3;
    S.GUI.LongPress2Window_s = 3;
    S.GUI.RewardWindowLeft_s = 0.2;
    S.GUI.RewardWindowRight_s = 1.5;
    S.GUI.RewardDelay_s = 0.1;
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
    S.GUI.OptoFrequency_Hz = 20;
    S.GUI.OptoPulseOn_ms = 10;
end

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
S.GUIMeta.ProbeMode.Style = 'checkbox';
S.GUIMeta.AssistMode.Style = 'checkbox';

S.GUIPanels.Session = session(:, 1)';
S.GUIPanels.Stimulus = stimulus(:, 1)';
S.GUIPanels.Timing = timing(:, 1)';
S.GUIPanels.Joystick = joystick(:, 1)';
S.GUIPanels.Assist = assist(:, 1)';
S.GUIPanels.Reward = reward(:, 1)';
S.GUIPanels.ITI = iti(:, 1)';
S.GUIPanels.Opto = opto(:, 1)';
S.GUIPanels.Probe = probe(:, 1)';
S.GUIPanels.Chemo = chemo(:, 1)';

S.ConfigVersion = 15;
end
