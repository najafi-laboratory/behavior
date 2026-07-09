numPresses = 0;
for trial = 1:SessionData.nTrials
    if ~isnan(SessionData.RawEvents.Trial{1, trial}.States.Reward(1))
        numPresses = numPresses + 1;
    end
end
disp(numPresses)