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

% Keep parameter groups compact so GUI panels match task structure.
session = {'MaxTrials', 1000; 'PressMode', 2; 'TrialMode', 4; 'BlockLength', 30; 'BlockLengthEdge', 5; 'ProbeMode', 0; 'ProbeFraction', 0.2; 'ProbeZeroEdgeTrials', 5};
stimulus = {'TimingMode', 2; 'SensoryCueMode', 3; 'SensoryCueDuration_s', 0.2; 'UseGeneratedGrating', 1};
audio = {'AudioStimFreq_Hz', 11025; 'AudioStimVolume', 0.02; 'AudioSamplingRate_Hz', 48000; 'AudioAttenuation_dB', -35; 'AudioRamp_ms', 1};
timing = {'ShortDelay_s', 0.5; 'LongDelay_s', 1; 'Press1Window_s', 2; 'ShortPress2Window_s', 5; 'LongPress2Window_s', 5};
joystick = {'PressThreshold', 0.7; 'RetractThreshold', 0.3; 'ServoInPos', 1638; 'ServoOutPos', 50; 'ServoMoveDelay_s', 0.05; 'ServoReturnTimeout_s', 1; 'AssistMode', 1; 'AssistFraction', 0.3};
reward = {'RewardWindowLeft_s', 0.1; 'RewardMaximumWindow_s', 0.5; 'RewardWindowRight_s', 1.5; 'PreRewardDelay_s', 0.5; 'PostRewardDelay_s', 1; 'TotalRewardDuration_s', 1; 'RewardMode', 1; 'RewardAmount_uL', 20; 'ShortRewardAmount_uL', 3; 'LongRewardAmount_uL', 3};
iti = {'ITIMode', 2; 'ManualITI_s', 1; 'ITIMin_s', 5; 'ITIMax_s', 7; 'ITIMean_s', 6; 'PunishITIMode', 2; 'ManualPunishITI_s', 0; 'PunishITIMin_s', 5; 'PunishITIMax_s', 7; 'PunishITIMean_s', 6};
manipulation = {'OptoMode', 0; 'OptoFraction', 0.35; 'OptoZeroEdgeTrials', 5; 'EnableOptoSensoryCue1', 1; 'EnableOptoDelay', 1; 'EnableOptoPreRewardDelay', 1; 'EnableOptoPostReward', 1; 'OptoFrequency_Hz', 50; 'OptoPulseOn_ms', 10; 'ChemoMode', 0};

groups = {session, stimulus, audio, timing, joystick, reward, iti, manipulation};
parameterNames = vertcat(session(:, 1), stimulus(:, 1), audio(:, 1), timing(:, 1), joystick(:, 1), reward(:, 1), iti(:, 1), manipulation(:, 1));

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

% Configure GUI widget types and menu labels.
S.GUIMeta = struct;
S.GUIPanels = struct;
S.GUIMeta.PressMode.Style = 'popupmenu';
S.GUIMeta.PressMode.String = {'Single Press', 'Double Press'};
S.GUIMeta.TrialMode.Style = 'popupmenu';
S.GUIMeta.TrialMode.String = {'All Short', 'All Long', 'Blocks Short First', 'Blocks Long First'};
S.GUIMeta.TimingMode.Style = 'popupmenu';
S.GUIMeta.TimingMode.String = {'Visual Guided', 'Self Timed'};
S.GUIMeta.SensoryCueMode.Style = 'popupmenu';
S.GUIMeta.SensoryCueMode.String = {'Visual only', 'Audio only', 'Audio + visual'};
S.GUIMeta.ChemoMode.Style = 'checkbox';
S.GUIMeta.UseGeneratedGrating.Style = 'checkbox';
S.GUIMeta.RewardMode.Style = 'popupmenu';
S.GUIMeta.RewardMode.String = {'Same Reward', 'Different Reward'};
S.GUIMeta.ITIMode.Style = 'popupmenu';
S.GUIMeta.ITIMode.String = {'Manual', 'Exponential'};
S.GUIMeta.PunishITIMode.Style = 'popupmenu';
S.GUIMeta.PunishITIMode.String = {'Manual', 'Exponential'};
S.GUIMeta.OptoMode.Style = 'checkbox';
S.GUIMeta.EnableOptoSensoryCue1.Style = 'checkbox';
S.GUIMeta.EnableOptoDelay.Style = 'checkbox';
S.GUIMeta.EnableOptoPreRewardDelay.Style = 'checkbox';
S.GUIMeta.EnableOptoPostReward.Style = 'checkbox';
S.GUIMeta.ProbeMode.Style = 'checkbox';
S.GUIMeta.AssistMode.Style = 'checkbox';

% Panel counts: Session 8, Stimulus 4, Audio 5, Timing 5, Joystick 8,
% Reward 10, ITI 10, Manipulation 10.
% This order gives balanced stock Bpod GUI columns after panel reversal.
S.GUIPanels.Joystick = joystick(:, 1)';
S.GUIPanels.Timing = timing(:, 1)';
S.GUIPanels.Reward = reward(:, 1)';
S.GUIPanels.Stimulus = stimulus(:, 1)';
S.GUIPanels.Audio = audio(:, 1)';
S.GUIPanels.ITI = iti(:, 1)';
S.GUIPanels.Session = session(:, 1)';
S.GUIPanels.Manipulation = manipulation(:, 1)';

end
