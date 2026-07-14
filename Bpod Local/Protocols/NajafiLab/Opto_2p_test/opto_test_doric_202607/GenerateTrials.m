function itiValues = GenerateTrials(S, nTrials)
% Generate ITI values from the current GUI parameters.
mode = S.GUI.ITIMode;
manualValue = S.GUI.ManualITI_s;
minimum = S.GUI.ITIMin_s;
maximum = S.GUI.ITIMax_s;
meanValue = S.GUI.ITIMean_s;
if mode == 1
    itiValues = repmat(manualValue, 1, nTrials);
    return
end
if minimum == maximum
    itiValues = repmat(minimum, 1, nTrials);
    return
end
upperProbability = exp(-minimum / meanValue);
lowerProbability = exp(-maximum / meanValue);
itiValues = -meanValue * log(upperProbability - rand(1, nTrials) * (upperProbability - lowerProbability));
end
