classdef PostProcess
    methods

function [BpodSystem] = SaveProcessedSessionData( ...
        obj, BpodSystem, VisStim, GrayPerturbISI)
    TrialType = BpodSystem.Data.TrialTypes(end);
    [trial_states, trial_events] = GetTrialStatesEvents(obj, BpodSystem);
    [trial_outcome] = GetOutcome(obj, trial_states);
    [trial_seq, trial_prepost] = GetVisStimSeq(obj, VisStim);
    [trial_isi] = GetISI(obj, VisStim);
    BpodSystem.Data.ProcessedSessionData(end+1) = {struct( ...
        trial_outcome = trial_outcome, ...
        trial_seq = trial_seq, ...
        trial_isi = trial_isi, ...
        trial_prepost = trial_prepost)};
end


function [trial_states, trial_events] = GetTrialStatesEvents( ...
        obj, BpodSystem)
    trial_states = BpodSystem.Data.RawEvents.Trial{end}.States;
    trial_events = BpodSystem.Data.RawEvents.Trial{end}.Events;
end


function [trial_outcome] = GetOutcome( ...
        obj, trial_states)
    if ( isfield(trial_states, 'Passive_VisualStim') && ...
            ~isnan(trial_states.Passive_VisualStim(1)))
        trial_outcome = 'Passive';
    elseif ( isfield(trial_states, 'ChangingMindReward') && ...
            ~isnan(trial_states.ChangingMindReward(1)))
        trial_outcome = 'ChangingMindReward';
    elseif ( isfield(trial_states, 'Punish') && ...
            ~isnan(trial_states.Punish(1)))
        trial_outcome = 'Punish';
    elseif ( isfield(trial_states, 'Reward') && ...
            ~isnan(trial_states.Reward(1)))
        trial_outcome = 'Reward';
    elseif ( isfield(trial_states, 'PunishNaive') && ...
            ~isnan(trial_states.PunishNaive(1)))
        trial_outcome = 'PunishNaive';
    elseif ( isfield(trial_states, 'RewardNaive') && ...
            ~isnan(trial_states.RewardNaive(1)))
        trial_outcome = 'RewardNaive';
    elseif ( isfield(trial_states, 'WrongInitiation') && ...
            ~isnan(trial_states.WrongInitiation(1)))
        trial_outcome = 'WrongInitiation';
    elseif ( isfield(trial_states, 'DidNotChoose') && ...
            ~isnan(trial_states.DidNotChoose(1)))
        trial_outcome = 'DidNotChoose';
    else
        trial_outcome = 'Others';
    end
end


function [trial_isi] = GetISI( ...
        obj, VisStim)
    % trial_isi = struct( ...
    %     PreISI = VisStim.PreISIinfo, ...
    %     PostISI = VisStim.PostISIinfo, ...
    %     ExtraISI = VisStim.ExtraISIinfo, ...
    %     PostMeanISI = VisStim.ProcessedData.PostMeanISI);
    trial_isi = struct( ...
        PreISI = VisStim.PreISIinfo, ...
        PostISI = VisStim.PostISIinfo, ...
        PostMeanISI = VisStim.ProcessedData.PostMeanISI);    
end


function [trial_seq, trial_prepost] = GetVisStimSeq( ...
        obj, VisStim)
    trial_seq = VisStim.ProcessedData.Seq;
    trial_prepost = VisStim.ProcessedData.PrePost;
end


    end
end