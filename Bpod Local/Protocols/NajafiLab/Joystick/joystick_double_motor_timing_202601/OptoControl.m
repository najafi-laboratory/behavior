function output = OptoControl(action, S, varargin)
% Generate opto trial tags and plotting metadata.
switch action
    case 'trials'
        % Draw opto trial types only when opto mode is enabled.
        nTrials = round(S.GUI.MaxTrials);
        output = zeros(1, nTrials);
        if S.GUI.OptoMode
            nOpto = min(nTrials, max(0, round(S.GUI.OptoFraction * nTrials)));
            indices = randperm(nTrials, nOpto);
            output(indices) = randi(3, 1, nOpto);
        end
    case 'actions'
        % Return state-machine output actions for the selected opto type.
        optoType = varargin{1};
        if ~ismember(optoType, 0:3)
            error('Opto trial type must be 0, 1, 2, or 3.')
        end
        output.Enabled = optoType > 0;
        output.TrialType = optoType;
        output.OutputActions = {};
    case 'display'
        % Keep labels and colors centralized for plotting.
        output.Labels = {'Off', 'Type 1', 'Type 2', 'Type 3'};
        output.Colors = [0.65 0.65 0.65; 0.49 0.18 0.56; 0.93 0.69 0.13; 0.3 0.75 0.93];
    otherwise
        error('Unknown opto action: %s', action)
end
end
