function probeTypes = GenerateProbeTrials(S, blockTypes)
% Generate tag-only probe trial types: 0 off, 1 stimulus only, 2 servo only.
nTrials = round(S.GUI.MaxTrials);
probeTypes = zeros(1, nTrials);
if S.GUI.TrainingMode == 1 || ~S.GUI.ProbeMode
    return
end

eligible = eligibleTrials(S, blockTypes);
nProbe = min(numel(eligible), max(0, round(S.GUI.ProbeFraction * numel(eligible))));
if nProbe == 0
    return
end

indices = eligible(randperm(numel(eligible), nProbe));
probeTypes(indices) = randi(2, 1, nProbe);
end

function eligible = eligibleTrials(S, blockTypes)
nTrials = numel(blockTypes);
edge = max(0, round(S.GUI.ProbeZeroEdgeTrials));
blocked = false(1, nTrials);
blockStarts = [1 find(diff(blockTypes) ~= 0) + 1];
blockEnds = [blockStarts(2:end) - 1 nTrials];

for i = 1:numel(blockStarts)
    firstEdge = blockStarts(i):min(blockEnds(i), blockStarts(i) + edge - 1);
    lastEdge = max(blockStarts(i), blockEnds(i) - edge + 1):blockEnds(i);
    blocked(firstEdge) = true;
    blocked(lastEdge) = true;
end
blocked(blockStarts(1):blockEnds(1)) = true;
eligible = find(~blocked);
end
