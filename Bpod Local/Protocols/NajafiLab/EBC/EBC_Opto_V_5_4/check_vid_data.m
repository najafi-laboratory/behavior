




SessionData.vidTime;

SessionData.vidMetadata;



vidTime_diff = diff(SessionData.vidTime);
numMissingFrames = SessionData.TriggerPulseCount - SessionData.FramesAcquired;

[largestDiffs, idx_largestDiffs] = maxk(vidTime_diff, numMissingFrames);  


% where(time_diff == max(time_diff))
% where(time_diff == max(time_diff))



[~, idx] = max(time_diff)