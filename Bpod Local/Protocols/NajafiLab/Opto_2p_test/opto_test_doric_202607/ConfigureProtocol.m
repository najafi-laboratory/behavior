function S = ConfigureProtocol(BpodSystem)
% Prepare the small parameter set needed by this opto timing test.
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

session = {'PreStimDelay_s', 1; 'ImageDuration_s', 0.5; 'ImageInterval_s', 0.5; 'PostStimDelay_s', 1};
iti = {'ITIMode', 2; 'ManualITI_s', 1; 'ITIMin_s', 3; 'ITIMax_s', 5; 'ITIMean_s', 4};
opto = {'OptoMode', 1; 'OptoFraction', 0.5; 'EnableOptoPreStimDelay', 0; 'EnableOptoInterval', 1; 'EnableOptoStim', 0; 'EnableOptoPostStimDelay', 0; 'LaserTriggerMode', 1; 'LaserDuration_s', 0.1; 'OptoFrequency_Hz', 50; 'OptoPulseOn_ms', 10};

groups = {session, iti, opto};
parameterNames = vertcat(session(:, 1), iti(:, 1), opto(:, 1));
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

S.GUIMeta = struct;
S.GUIPanels = struct;
S.GUIMeta.ITIMode.Style = 'popupmenu';
S.GUIMeta.ITIMode.String = {'Manual', 'Exponential'};
S.GUIMeta.OptoMode.Style = 'checkbox';
S.GUIMeta.EnableOptoPreStimDelay.Style = 'checkbox';
S.GUIMeta.EnableOptoInterval.Style = 'checkbox';
S.GUIMeta.EnableOptoStim.Style = 'checkbox';
S.GUIMeta.EnableOptoPostStimDelay.Style = 'checkbox';
S.GUIMeta.LaserTriggerMode.Style = 'popupmenu';
S.GUIMeta.LaserTriggerMode.String = {'Full epoch gate', 'Fixed duration from onset'};

S.GUIPanels.Sessions = session(:, 1)';
S.GUIPanels.ITI = iti(:, 1)';
S.GUIPanels.Opto = opto(:, 1)';
S.ConfigVersion = 1;
end
