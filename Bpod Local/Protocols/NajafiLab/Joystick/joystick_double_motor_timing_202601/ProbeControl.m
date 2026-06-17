function output = ProbeControl(action, S, varargin)
% Generate probe trial tags and plotting metadata.
switch action
    case 'trials'
        % Draw probe trial types only when probe mode is enabled.
        trialTypes = [];
        if ~isempty(varargin)
            trialTypes = varargin{1};
        end

        nTrials = round(S.GUI.MaxTrials);
        output = zeros(1, nTrials);
        if S.GUI.ProbeMode
            eligible = eligibleTagTrials(S, nTrials, trialTypes);
            nProbe = min(numel(eligible), max(0, round(S.GUI.ProbeFraction * numel(eligible))));
            if nProbe > 0
                indices = eligible(randperm(numel(eligible), nProbe));
                output(indices) = randi(2, 1, nProbe);
            end
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

function eligible = eligibleTagTrials(S, nTrials, trialTypes)
edge = max(0, round(S.GUI.ProbeZeroEdgeTrials));
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
