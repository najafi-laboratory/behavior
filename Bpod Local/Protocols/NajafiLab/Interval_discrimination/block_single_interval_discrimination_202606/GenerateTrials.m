function [trialTypes, blockTypes, isiValues, itiValues, punishITIValues, blockStarts, blockEnds] = GenerateTrials(S)
% Generate probability blocks, trial types, ISI values, and ITI values.
nTrials = round(S.GUI.MaxTrials);

[blockTypes, blockStarts, blockEnds] = generateBlocks(S, nTrials);
trialTypes = zeros(1, nTrials);
for block = 1:numel(blockStarts)
    trials = blockStarts(block):blockEnds(block);
    trialTypes(trials) = sampleBlockTrials(S, blockTypes(trials(1)), numel(trials));
end

isiValues = zeros(1, nTrials);
for trial = 1:nTrials
    if trialTypes(trial) == 1
        isiValues(trial) = sampleValue(S.GUI.ShortISIMode, S.GUI.ShortISIFixed_s, S.GUI.ShortISIMin_s, S.GUI.ShortISIMax_s);
    else
        isiValues(trial) = sampleValue(S.GUI.LongISIMode, S.GUI.LongISIFixed_s, S.GUI.LongISIMin_s, S.GUI.LongISIMax_s);
    end
end
itiValues = generateITI(S.GUI.ITIMode, S.GUI.ManualITI_s, S.GUI.ITIMin_s, S.GUI.ITIMax_s, S.GUI.ITIMean_s, nTrials);
punishITIValues = generateITI(S.GUI.PunishITIMode, S.GUI.ManualPunishITI_s, S.GUI.PunishITIMin_s, S.GUI.PunishITIMax_s, S.GUI.PunishITIMean_s, nTrials);
end

function [blockTypes, blockStarts, blockEnds] = generateBlocks(S, nTrials)
blockTypes = zeros(1, nTrials);
blockStarts = [];
blockEnds = [];
blockIndex = 1;
blockType = blockTypeForIndex(S, blockIndex, []);
firstTrial = 1;
minimumLength = max(1, round(S.GUI.BlockLength - S.GUI.BlockMargin));
maximumLength = max(minimumLength, round(S.GUI.BlockLength + S.GUI.BlockMargin));

while firstTrial <= nTrials
    blockLength = randi([minimumLength maximumLength]);
    lastTrial = min(nTrials, firstTrial + blockLength - 1);
    blockTypes(firstTrial:lastTrial) = blockType;
    blockStarts(end + 1) = firstTrial;
    blockEnds(end + 1) = lastTrial;
    blockIndex = blockIndex + 1;
    blockType = blockTypeForIndex(S, blockIndex, blockType);
    firstTrial = lastTrial + 1;
end
end

function blockType = blockTypeForIndex(S, blockIndex, previousType)
warmupBlocks = max(0, round(S.GUI.WarmupBlockNum));
if S.GUI.BlockNum == 1 || blockIndex <= warmupBlocks
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

function value = sampleValue(mode, fixedValue, minimum, maximum)
if mode == 1
    value = fixedValue;
else
    value = minimum + rand * (maximum - minimum);
end
end

function values = generateITI(mode, manualValue, minimum, maximum, meanValue, nTrials)
if mode == 1
    values = repmat(manualValue, 1, nTrials);
    return
end
if minimum == maximum
    values = repmat(minimum, 1, nTrials);
    return
end

upperProbability = exp(-minimum / meanValue);
lowerProbability = exp(-maximum / meanValue);
values = -meanValue * log(upperProbability - rand(1, nTrials) * (upperProbability - lowerProbability));
end
