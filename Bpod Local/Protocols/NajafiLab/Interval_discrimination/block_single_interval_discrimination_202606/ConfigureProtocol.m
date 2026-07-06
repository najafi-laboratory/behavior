function S = ConfigureProtocol(BpodSystem)
% Prepare GUI defaults for the short/long AV interval protocol.
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

session = {'MaxTrials', 1000; 'TrainingMode', 2; 'Contingency', 1};
blocks = {'BlockNum', 1; 'WarmupBlockNum', 1; 'BlockLength', 30; 'BlockMargin', 3; 'BlockEdgeTrials', 4; 'MostFraction', 0.8};
stimulus = {'StimulusMode', 3; 'UseSavedImage', 0; 'PreStimDelay_s', 0.1; 'GratingDuration_s', 0.2};
isi = {'ShortISIMode', 1; 'ShortISIFixed_s', 0.5; 'ShortISIMin_s', 0.5; 'ShortISIMax_s', 0.7; 'LongISIMode', 1; 'LongISIFixed_s', 2.5; 'LongISIMin_s', 2.3; 'LongISIMax_s', 2.5};
audio = {'AudioStimFreq_Hz', 11025; 'AudioStimVolume', 0.1; 'AudioSamplingRate_Hz', 44100; 'AudioAttenuation_dB', -35; 'AudioRamp_ms', 1};
opto = {'OptoMode', 1; 'OptoFraction', 0.35; 'OptoZeroEdgeTrials', 5; 'OptoEarlyTrials', 5; 'OptoTriggerType', 1; 'OptoTriggerMode', 1; 'EnableOptoStimulus', 1; 'EnableOptoSpoutInDelay', 0; 'EnableOptoChoice', 0; 'EnableOptoPreOutcome', 0; 'EnableOptoReward', 0; 'EnableOptoPostReward', 0; 'EnableOptoPunishITI', 0};
probe = {'ProbeMode', 0; 'ProbeFraction', 0.1; 'ProbeZeroEdgeTrials', 5};
choice = {'SpoutInDelay_s', 0.2; 'ChoiceWindow_s', 5; 'AllowChangeMind', 0; 'ChangeMindWindow_s', 0.5};
reward = {'PreOutcomeDelay_s', 0.1; 'PostRewardDelay_s', 1.5; 'LeftRewardAmount_uL', 6; 'RightRewardAmount_uL', 6};
servo = {'CurrentSpoutPosition', 1; 'RightServoInPos', 1220; 'LeftServoInPos', 1810; 'ServoDeflection', 90; 'ServoVelocity', 1; 'ServoMoveDelay_s', 0.1; 'ServoReturnTimeout_s', 1};
iti = {'ITIMode', 2; 'ManualITI_s', 1; 'ITIMin_s', 3; 'ITIMax_s', 6; 'ITIMean_s', 4.5; 'PunishITIMode', 2; 'ManualPunishITI_s', 0; 'PunishITIMin_s', 3; 'PunishITIMax_s', 7; 'PunishITIMean_s', 5};
chemo = {'ChemoMode', 0};

groups = {session, blocks, stimulus, isi, audio, opto, probe, choice, reward, servo, iti, chemo};
parameterNames = vertcat(session(:, 1), blocks(:, 1), stimulus(:, 1), isi(:, 1), audio(:, 1), opto(:, 1), probe(:, 1), choice(:, 1), reward(:, 1), servo(:, 1), iti(:, 1), chemo(:, 1));

if isfield(S.GUI, 'RewardDelay_s') && ~isfield(S.GUI, 'PreOutcomeDelay_s')
    S.GUI.PreOutcomeDelay_s = S.GUI.RewardDelay_s;
end
if isfield(S.GUI, 'PreRewardDelay_s') && ~isfield(S.GUI, 'PreOutcomeDelay_s')
    S.GUI.PreOutcomeDelay_s = S.GUI.PreRewardDelay_s;
end
if isfield(S.GUI, 'EnableOptoPreReward') && ~isfield(S.GUI, 'EnableOptoPreOutcome')
    S.GUI.EnableOptoPreOutcome = S.GUI.EnableOptoPreReward;
end

for groupIndex = 1:numel(groups)
    group = groups{groupIndex};
    for parameterIndex = 1:size(group, 1)
        name = group{parameterIndex, 1};
        if ~isfield(S.GUI, name)
            S.GUI.(name) = group{parameterIndex, 2};
        end
    end
end
if S.GUI.OptoMode < 1
    S.GUI.OptoMode = 1;
end

unusedParameters = setdiff(fieldnames(S.GUI), parameterNames);
if ~isempty(unusedParameters)
    S.GUI = rmfield(S.GUI, unusedParameters);
end

S.GUIMeta = struct;
S.GUIPanels = struct;
S.GUIMeta.TrainingMode.Style = 'popupmenu';
S.GUIMeta.TrainingMode.String = {'Naive', 'Trained'};
S.GUIMeta.Contingency.Style = 'popupmenu';
S.GUIMeta.Contingency.String = {'Short-left, long-right', 'Short-right, long-left'};
S.GUIMeta.StimulusMode.Style = 'popupmenu';
S.GUIMeta.StimulusMode.String = {'Visual only', 'Audio only', 'Audio + visual'};
S.GUIMeta.UseSavedImage.Style = 'checkbox';
S.GUIMeta.BlockNum.Style = 'popupmenu';
S.GUIMeta.BlockNum.String = {'50/50 only', '50/50 then left/right', '50/50, left, right'};
S.GUIMeta.ShortISIMode.Style = 'popupmenu';
S.GUIMeta.ShortISIMode.String = {'Fixed', 'Uniform random'};
S.GUIMeta.LongISIMode.Style = 'popupmenu';
S.GUIMeta.LongISIMode.String = {'Fixed', 'Uniform random'};
S.GUIMeta.OptoMode.Style = 'popupmenu';
S.GUIMeta.OptoMode.String = {'No opto', 'Random', 'Early trials in every block', 'Early trials in alternating block groups'};
S.GUIMeta.OptoTriggerType.Style = 'popupmenu';
S.GUIMeta.OptoTriggerType.String = {'Manual', 'Triggered', 'Gated'};
S.GUIMeta.OptoTriggerMode.Style = 'popupmenu';
S.GUIMeta.OptoTriggerMode.String = {'Pause', 'Continue', 'Restart', 'Uninterrupted'};
S.GUIMeta.EnableOptoStimulus.Style = 'checkbox';
S.GUIMeta.EnableOptoSpoutInDelay.Style = 'checkbox';
S.GUIMeta.EnableOptoChoice.Style = 'checkbox';
S.GUIMeta.EnableOptoPreOutcome.Style = 'checkbox';
S.GUIMeta.EnableOptoReward.Style = 'checkbox';
S.GUIMeta.EnableOptoPostReward.Style = 'checkbox';
S.GUIMeta.EnableOptoPunishITI.Style = 'checkbox';
S.GUIMeta.ChemoMode.Style = 'checkbox';
S.GUIMeta.ProbeMode.Style = 'checkbox';
S.GUIMeta.AllowChangeMind.Style = 'checkbox';
S.GUIMeta.ITIMode.Style = 'popupmenu';
S.GUIMeta.ITIMode.String = {'Manual', 'Exponential'};
S.GUIMeta.PunishITIMode.Style = 'popupmenu';
S.GUIMeta.PunishITIMode.String = {'Manual', 'Exponential'};
S.GUIMeta.CurrentSpoutPosition.Style = 'popupmenu';
S.GUIMeta.CurrentSpoutPosition.String = {'Neutral', 'Leftwards', 'Rightwards'};

S.GUIPanels.Session = session(:, 1)';
S.GUIPanels.Blocks = blocks(:, 1)';
S.GUIPanels.Stimulus = stimulus(:, 1)';
S.GUIPanels.Audio = audio(:, 1)';
S.GUIPanels.ISI = isi(:, 1)';
S.GUIPanels.Opto = opto(:, 1)';
S.GUIPanels.Chemo = chemo(:, 1)';
S.GUIPanels.Probe = probe(:, 1)';
S.GUIPanels.Choice = choice(:, 1)';
S.GUIPanels.Reward = reward(:, 1)';
S.GUIPanels.Servo = servo(:, 1)';
S.GUIPanels.ITI = iti(:, 1)';
S.ConfigVersion = 4;
end
