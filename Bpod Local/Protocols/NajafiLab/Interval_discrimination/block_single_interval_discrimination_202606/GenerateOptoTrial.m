function optoType = GenerateOptoTrial(S, blockTypes, blockStarts, blockEnds, trial)
% Generate one opto column using current GUI settings.
optoType = zeros(7, 1);
if S.GUI.TrainingMode == 1
    return
end
mode = round(S.GUI.OptoMode);
if mode == 1
    return
end

enabledPeriods = [S.GUI.EnableOptoStimulus; S.GUI.EnableOptoSpoutInDelay; S.GUI.EnableOptoSpoutIn; S.GUI.EnableOptoPreOutcome; S.GUI.EnableOptoReward; S.GUI.EnableOptoPostReward; S.GUI.EnableOptoPunishITI] ~= 0;
if ~any(enabledPeriods)
    return
end

if nargin < 3 || isempty(blockStarts) || isempty(blockEnds)
    [blockStarts, blockEnds] = blockEdges(blockTypes);
end
warmupBlocks = min(numel(blockStarts), leadingFiftyFiftyBlocks(S));
block = find(trial >= blockStarts & trial <= blockEnds, 1);
if isempty(block) || block <= warmupBlocks
    return
end

switch mode
    case 2
        isOptoTrial = isRandomOptoTrial(S, blockStarts(block), blockEnds(block), trial);
    case 3
        isOptoTrial = ismember(trial, earlyTrialsInBlock(S, blockStarts(block), blockEnds(block)));
    case 4
        groupSize = max(1, round(S.GUI.BlockNum));
        groupIndex = floor((block - warmupBlocks - 1) / groupSize);
        isOptoTrial = mod(groupIndex, 2) == 1 && ismember(trial, earlyTrialsInBlock(S, blockStarts(block), blockEnds(block)));
    otherwise
        isOptoTrial = false;
end

if isOptoTrial
    optoType(enabledPeriods) = 1;
end
end

function count = leadingFiftyFiftyBlocks(S)
if S.GUI.BlockNum == 1
    count = inf;
else
    count = 1 + max(0, round(S.GUI.WarmupBlockNum));
end
end

function isOpto = isRandomOptoTrial(S, blockStart, blockEnd, trial)
edge = max(0, round(S.GUI.OptoZeroEdgeTrials));
inFirstEdge = trial <= min(blockEnd, blockStart + edge - 1);
inLastEdge = trial >= max(blockStart, blockEnd - edge + 1);
isOpto = ~inFirstEdge && ~inLastEdge && rand < S.GUI.OptoFraction;
end

function trials = earlyTrialsInBlock(S, blockStart, blockEnd)
count = max(0, round(S.GUI.OptoEarlyTrials));
trials = blockStart:min(blockEnd, blockStart + count - 1);
end

function [blockStarts, blockEnds] = blockEdges(blockTypes)
nTrials = numel(blockTypes);
blockStarts = [1 find(diff(blockTypes) ~= 0) + 1];
blockEnds = [blockStarts(2:end) - 1 nTrials];
end
