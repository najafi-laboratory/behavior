





nTrials = SessionData.nTrials;
vidStartTime = [];
for trial = 1:nTrials
    vidStartTime = [vidStartTime SessionData.RawEvents.Trial{1, trial}.Data.FECTimes(1)];

end