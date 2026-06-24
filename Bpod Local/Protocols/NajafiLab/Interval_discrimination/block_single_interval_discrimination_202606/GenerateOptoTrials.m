function optoTypes = GenerateOptoTrials(S, blockTypes, blockStarts, blockEnds)
% Allocate or generate period-wise opto tags: rows are stimulus, choice, reward.
nTrials = round(S.GUI.MaxTrials);
optoTypes = zeros(3, nTrials);
for trial = 1:nTrials
    optoTypes(:, trial) = GenerateOptoTrial(S, blockTypes, blockStarts, blockEnds, trial);
end
end
