function [trialTypes, itiValues, punishITIValues] = GenerateTrials(S)
% Generate trial types and ITI values from current GUI parameters.
nTrials = round(S.GUI.MaxTrials);
itiValues = generateITI(S, nTrials, false);
punishITIValues = generateITI(S, nTrials, true);

if S.GUI.TrialMode == 1
    trialTypes = ones(1, nTrials);
    return
end

if S.GUI.TrialMode == 2
    trialTypes = 2 * ones(1, nTrials);
    return
end

trialTypes = zeros(1, nTrials);
trialType = S.GUI.TrialMode - 2;
firstTrial = 1;
minimumLength = max(1, round(S.GUI.BlockLength - S.GUI.BlockLengthEdge));
maximumLength = max(minimumLength, round(S.GUI.BlockLength + S.GUI.BlockLengthEdge));

% Blocks alternate short and long with randomized block length.
while firstTrial <= nTrials
    blockLength = randi([minimumLength maximumLength]);
    lastTrial = min(nTrials, firstTrial + blockLength - 1);
    trialTypes(firstTrial:lastTrial) = trialType;
    trialType = 3 - trialType;
    firstTrial = lastTrial + 1;
end
end

function values = generateITI(S, nTrials, punish)
% Draw manual or truncated exponential ITI for each trial.
if punish
    mode = S.GUI.PunishITIMode;
    manualValue = S.GUI.ManualPunishITI_s;
    minimum = S.GUI.PunishITIMin_s;
    maximum = S.GUI.PunishITIMax_s;
    meanValue = S.GUI.PunishITIMean_s;
else
    mode = S.GUI.ITIMode;
    manualValue = S.GUI.ManualITI_s;
    minimum = S.GUI.ITIMin_s;
    maximum = S.GUI.ITIMax_s;
    meanValue = S.GUI.ITIMean_s;
end
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
