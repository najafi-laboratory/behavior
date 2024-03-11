classdef PostProcess
    methods

function [BpodSystem] = SaveProcessedSessionData( ...
        obj, BpodSystem, VisStim, GrayPerturbISI)
    TrialType = BpodSystem.Data.TrialTypes(end);
    [trial_states, trial_events] = GetTrialStatesEvents(obj, BpodSystem);
    [trial_outcome] = GetOutcome(obj, trial_states);
    [trial_iti] = GetITI(obj, trial_states);
    [trial_avsync] = GetAVsync(obj, trial_events);
    [trial_licking, trial_reaction] = GetLicking(obj, trial_states, trial_events, TrialType);
    [trial_seq, trial_prepost] = GetVisStimSeq(obj, VisStim);
    [trial_isi] = GetISI(obj, VisStim);
    [trial_choice] = GetChoice( ...
        obj, TrialType, trial_outcome, GrayPerturbISI);
    [trial_com] = GetChangOfMind( ...
        obj, TrialType, trial_outcome);
    BpodSystem.Data.ProcessedSessionData(end+1) = {struct( ...
        trial_outcome = trial_outcome, ...
        trial_iti = trial_iti, ...
        trial_avsync = trial_avsync, ...
        trial_licking = trial_licking, ...
        trial_reaction = trial_reaction, ...
        trial_seq = trial_seq, ...
        trial_isi = trial_isi, ...
        trial_choice = trial_choice, ...
        trial_com = trial_com, ...
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


function [trial_iti] = GetITI( ...
        obj, trial_states)
    iti = trial_states.ITI;
    trial_iti = iti(2) - iti(1);
end


function [trial_isi] = GetISI( ...
        obj, VisStim)
    trial_isi = struct( ...
        PreISI = VisStim.PreISIinfo, ...
        PostISI = VisStim.PostISIinfo, ...
        ExtraISI = VisStim.ExtraISIinfo);
end


function [trial_avsync] = GetAVsync( ...
        obj, trial_events)
    trial_avsync = nan;
    if (isfield(trial_events, 'BNC1High') && isfield(trial_events, 'BNC2High'))
        BNC1High = trial_events.BNC1High;
        BNC1High = reshape(BNC1High, [], 1);
        BNC2High = trial_events.BNC2High;
        BNC2High = reshape(BNC2High, [], 1);
        FirstGrating = BNC1High(1);
        FirstGratingAudioDiff = FirstGrating - BNC2High;
        FirstGratingAudioDiff_abs = abs(FirstGratingAudioDiff);
        [~, AudioStimStartIndex] = min(FirstGratingAudioDiff_abs);
        trial_avsync = FirstGrating - BNC2High(AudioStimStartIndex);
    end
end


function [trial_licking, trial_reaction] = GetLicking( ...
        obj, trial_states, trial_events, TrialType)
    trial_licking = [];
    trial_reaction = [];
    if (isfield(trial_states, 'VisStimTrigger') && ...
        ~isnan(trial_states.VisStimTrigger(2)))
        stim_start = trial_states.VisStimTrigger(2);
        licking_events = [];
        direction = [];
        correctness = [];
        if isfield(trial_events, 'Port1In')
            lick_left = reshape(trial_events.Port1In, [], 1) - stim_start;
            licking_events = [licking_events; lick_left];
            direction = [direction; zeros(size(lick_left))];
            if TrialType == 1
                correctness = [correctness; ones(size(lick_left))];
            else
                correctness = [correctness; zeros(size(lick_left))];
            end
        end
        if isfield(trial_events, 'Port3In')
            lick_right = reshape(trial_events.Port3In, [], 1) - stim_start;
            licking_events = [licking_events; lick_right];
            direction = [direction; ones(size(lick_right))];
            if TrialType == 2
                correctness = [correctness; ones(size(lick_right))];
            else
                correctness = [correctness; zeros(size(lick_right))];
            end
        end
        if ~isempty(licking_events)
            licking_events = reshape(licking_events, 1, []);
            correctness = reshape(correctness, 1, []);
            direction = reshape(direction, 1, []);
            lick = [licking_events; correctness; direction];
            trial_licking = [trial_licking, lick];
            lick = lick(:, lick(1,:) > 0);
            if size(lick, 2) > 0
                [~, reaction_idx] = min(lick(1,:));
                trial_reaction = [trial_reaction, reshape(lick(:,reaction_idx), 1, [])];
            end
        end
    end
end


function [trial_choice] = GetChoice( ...
        obj, TrialType, trial_outcome, GrayPerturbISI)
    choice = nan;
    if strcmp(trial_outcome, 'Reward')
        choice = TrialType - 1;
    elseif any(strcmp(trial_outcome, {'Punish', 'ChangingMindReward'}))
        choice = 2 - TrialType;
    end
    trial_choice = [GrayPerturbISI, choice];
end


function [trial_com] = GetChangOfMind( ...
        obj, TrialType, trial_outcome)
    trial_com = nan;
    if strcmp(trial_outcome, 'ChangingMindReward')
        trial_com = TrialType - 1;
    end
end


function [trial_seq, trial_prepost] = GetVisStimSeq( ...
        obj, VisStim)
    trial_seq = VisStim.ProcessedData.Seq;
    trial_prepost = VisStim.ProcessedData.PrePost;
end


    end
end