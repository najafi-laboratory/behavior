classdef Plotter
    methods


function UpdateSideOutcomePlot(obj, BpodSystem, TrialTypes, isEndOfTrial)
    Data = BpodSystem.Data;
    if 0
    if isfield(Data, 'nTrials')
        Outcomes = zeros(1,Data.nTrials);
        for x = 1:Data.nTrials
            if ~isnan(Data.RawEvents.Trial{x}.States.Reward(1))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp(['Outcome: Reward']);
                end
                Outcomes(x) = 1;    % draws green circle on outcome plot
            elseif ~isnan(Data.RawEvents.Trial{x}.States.Punish(1))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp(['Outcome: Punish']);
                end
                Outcomes(x) = 0;    % draws red circle on outcome plot
            elseif ~isnan(Data.RawEvents.Trial{x}.States.ExtraStimDurPostRew_Naive(1))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp(['Outcome: RewardNaive']);
                end
                Outcomes(x) = 1;    % draws green circle on outcome plot
            elseif ~isnan(Data.RawEvents.Trial{x}.States.PunishNaive(1))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp(['Outcome: PunishNaive']);
                end
                Outcomes(x) = 0;    % draws red circle on outcome plot
            elseif ~isnan(Data.RawEvents.Trial{x}.States.WrongInitiation(1))
                if (x == Data.nTrials && isEndOfTrial)
                    disp(['Outcome: WrongInitiation']);
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            elseif ~isnan(Data.RawEvents.Trial{x}.States.EarlyChoice(1))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp(['Outcome: EarlyChoice']);
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            elseif ~isnan(Data.RawEvents.Trial{x}.States.EarlyChoiceDurCenterLick(1))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp(['Outcome: EarlyChoiceDurCenterLick']);
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            elseif ~isnan(Data.RawEvents.Trial{x}.States.DidNotChoose(1))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp(['Outcome: DidNotChoose']);
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            elseif ~isnan(Data.RawEvents.Trial{x}.States.DidNotConfirm(1))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp(['Outcome: DidNotConfirm']);
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            elseif ~isnan(Data.RawEvents.Trial{x}.States.DidNotLickCenter(1))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp(['Outcome: DidNotLickCenter']);
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            else
                % this is 'catch-all' to indicate that none of the above outcomes
                % occured so we know that we need to find/add more outcomes to list
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured          
                    disp(['Outcome: Other']);
                end
                Outcomes(x) = 3;    % draws clear circle on outcome plot
            end
            if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                %disp(['Data.nTrials:', num2str(Data.nTrials)]);
                disp('-------------------------------------------------------'); % visual barrier for experimenter info
            end
        end
        SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'update',Data.nTrials+1,2-TrialTypes,Outcomes);
    else
        % if field nTrials doesn't exist in Data struct, then this is the first
        % trial so we only want SideOutcomePlot to update the trial choice case 
        Outcomes = zeros(1, 1);
        SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'update', 1,2-TrialTypes, Outcomes);
    end
    end
end


    end
end