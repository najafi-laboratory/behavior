function [trialTypes, blockTypes, blockStarts, blockEnds] = GenerateTrials(S)
% Generate probability blocks and short/long trial types.
nTrials = round(S.GUI.MaxTrials);

[blockTypes, blockStarts, blockEnds] = generateBlocks(S, nTrials);
trialTypes = zeros(1, nTrials);
for block = 1:numel(blockStarts)
    trials = blockStarts(block):blockEnds(block);
    trialTypes(trials) = sampleBlockTrials(S, blockTypes(trials(1)), numel(trials));
end

end

function [blockTypes, blockStarts, blockEnds] = generateBlocks(S, nTrials)
blockTypes = zeros(1, nTrials);
minimumLength = max(1, round(S.GUI.BlockLength - S.GUI.BlockMargin));
maximumLength = max(minimumLength, round(S.GUI.BlockLength + S.GUI.BlockMargin));
maxBlocks = ceil(nTrials / minimumLength);
blockStarts = zeros(1, maxBlocks);
blockEnds = zeros(1, maxBlocks);

blockIndex = 1;
blockType = blockTypeForIndex(S, blockIndex, []);
firstTrial = 1;
while firstTrial <= nTrials
    blockLength = randi([minimumLength maximumLength]);
    lastTrial = min(nTrials, firstTrial + blockLength - 1);
    blockTypes(firstTrial:lastTrial) = blockType;
    blockStarts(blockIndex) = firstTrial;
    blockEnds(blockIndex) = lastTrial;
    blockIndex = blockIndex + 1;
    blockType = blockTypeForIndex(S, blockIndex, blockType);
    firstTrial = lastTrial + 1;
end
blockStarts = blockStarts(1:blockIndex - 1);
blockEnds = blockEnds(1:blockIndex - 1);
end

function blockType = blockTypeForIndex(S, blockIndex, previousType)
if S.GUI.BlockNum == 1 || blockIndex <= leadingFiftyFiftyBlocks(S)
    blockType = 1;
    return
end
if S.GUI.BlockNum == 2
    if isempty(previousType) || previousType == 1
        blockType = randi([2 3]);
    else
        blockType = 5 - previousType;
    end
else
    if isempty(previousType)
        candidates = 1:3;
    else
        candidates = setdiff(1:3, previousType);
    end
    blockType = candidates(randi(numel(candidates)));
end
end

function count = leadingFiftyFiftyBlocks(S)
count = 1 + max(0, round(S.GUI.WarmupBlockNum));
end

function trialTypes = sampleBlockTrials(S, blockType, nTrials)
switch blockType
    case 1
        pLeft = 0.5;
    case 2
        pLeft = S.GUI.MostFraction;
    case 3
        pLeft = 1 - S.GUI.MostFraction;
end

trialTypes = 1 + (rand(1, nTrials) >= pLeft);
if blockType == 1
    return
end

edgeTrials = min(round(S.GUI.BlockEdgeTrials), floor(nTrials / 2));
majorityType = blockType - 1;
if edgeTrials > 0
    trialTypes(1:edgeTrials) = majorityType;
    trialTypes(end - edgeTrials + 1:end) = majorityType;
end
end
