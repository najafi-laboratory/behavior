classdef PostProcess
    methods

function [BpodSystem] = SaveProcessedSessionData( ...
        obj, BpodSystem, VisStim)
    [trial_seq] = GetVisStimSeq(obj, VisStim);
    [trial_isi] = GetISI(obj, VisStim);
    BpodSystem.Data.ProcessedSessionData(end+1) = {struct( ...
        trial_seq = trial_seq, trial_isi = trial_isi)};
end


function [trial_isi] = GetISI( ...
        obj, VisStim)
    trial_isi = VisStim.PreISIinfo;
end


function [trial_seq] = GetVisStimSeq( ...
        obj, VisStim)
    trial_seq = VisStim.ProcessedData.Seq;
end


    end
end