function optoTypes = GenerateOptoTrials(S, blockTypes, blockStarts, blockEnds)
% Allocate or generate period-wise opto tags: stimulus, choice, pre-reward, post-reward, punish ITI.
nTrials = round(S.GUI.MaxTrials);
optoTypes = zeros(5, nTrials);
for trial = 1:nTrials
    optoTypes(:, trial) = GenerateOptoTrial(S, blockTypes, blockStarts, blockEnds, trial);
end
end
