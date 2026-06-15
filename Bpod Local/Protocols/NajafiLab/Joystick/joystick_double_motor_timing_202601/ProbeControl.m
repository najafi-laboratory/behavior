function output = ProbeControl(action, S, varargin)
% Generate probe trial tags and plotting metadata.
switch action
    case 'trials'
        % Draw probe trial types only when probe mode is enabled.
        nTrials = round(S.GUI.MaxTrials);
        output = zeros(1, nTrials);
        if S.GUI.ProbeMode
            nProbe = min(nTrials, max(0, round(S.GUI.ProbeFraction * nTrials)));
            indices = randperm(nTrials, nProbe);
            output(indices) = randi(2, 1, nProbe);
        end
    case 'actions'
        % Return per-trial probe metadata for future state-machine hooks.
        probeType = varargin{1};
        if ~ismember(probeType, 0:2)
            error('Probe trial type must be 0, 1, or 2.')
        end
        output.Enabled = probeType > 0;
        output.TrialType = probeType;
        output.OutputActions = {};
    case 'display'
        % Keep labels and colors centralized for plotting.
        output.Labels = {'Off', 'Type 1', 'Type 2'};
        output.Colors = [0.65 0.65 0.65; 0.18 0.55 0.34; 0.82 0.33 0.45];
    otherwise
        error('Unknown probe action: %s', action)
end
end
