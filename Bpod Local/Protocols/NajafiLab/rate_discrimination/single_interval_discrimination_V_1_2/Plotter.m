classdef Plotter
    methods


function UpdateOutcomePlot(obj, BpodSystem, TrialTypes, isEndOfTrial)
    Data = BpodSystem.Data;
    if isfield(Data, 'nTrials')
        Outcomes = zeros(1,Data.nTrials);
        for x = 1:Data.nTrials
            if ( isfield(Data.RawEvents.Trial{x}.States, 'Passive_VisualStim') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.Passive_VisualStim(1)))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp('Outcome: Passive');
                end
                Outcomes(x) = 1;    % draws green circle on outcome plot
            elseif ( isfield(Data.RawEvents.Trial{x}.States, 'ChangingMindReward') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.ChangingMindReward(1)))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp('Outcome: ChangingMindReward');
                end
                Outcomes(x) = 0;    % draws red circle on outcome plot
            elseif ( isfield(Data.RawEvents.Trial{x}.States, 'WrongInitiation') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.WrongInitiation(1)))
                if (x == Data.nTrials && isEndOfTrial)
                    disp('Outcome: WrongInitiation');
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            elseif ( isfield(Data.RawEvents.Trial{x}.States, 'Reward') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.Reward(1)))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp('Outcome: Reward');
                end
                Outcomes(x) = 1;    % draws green circle on outcome plot
            elseif ( isfield(Data.RawEvents.Trial{x}.States, 'Punish') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.Punish(1)))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp('Outcome: Punish');
                end
                Outcomes(x) = 0;    % draws red circle on outcome plot
            elseif ( isfield(Data.RawEvents.Trial{x}.States, 'DidNotChoose') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.DidNotChoose(1)))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp('Outcome: DidNotChoose');
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            elseif ( isfield(Data.RawEvents.Trial{x}.States, 'PunishNaive') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.PunishNaive(1)))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp('Outcome: PunishNaive');
                end
                Outcomes(x) = 0;    % draws red circle on outcome plot
            elseif ( isfield(Data.RawEvents.Trial{x}.States, 'RewardNaive') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.RewardNaive(1)))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp('Outcome: RewardNaive');
                end
                Outcomes(x) = 1;    % draws green circle on outcome plot
            else
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured          
                    disp('Outcome: Other');
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            end
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                %disp(['Data.nTrials:', num2str(Data.nTrials)]);
                disp('-------------------------------------------------------'); % visual barrier for experimenter info
            end
        end
        TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update', Data.nTrials+1, TrialTypes, Outcomes);
    else
        Outcomes = zeros(1, 1);
        TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update', 1, TrialTypes, Outcomes);
        yticks([1 2])
        yticklabels({'left','right'})
    end
end


    end
end