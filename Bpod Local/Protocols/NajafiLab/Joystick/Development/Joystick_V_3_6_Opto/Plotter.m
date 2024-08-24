classdef Plotter
    methods


function UpdateOutcomePlot(obj, BpodSystem, TrialTypes, OptoTrialTypes, ProbeTrialTypes, isEndOfTrial)
    Data = BpodSystem.Data;
    if isfield(Data, 'nTrials')
        Outcomes = zeros(1,Data.nTrials);
        for x = 1:Data.nTrials
            if ( isfield(Data.RawEvents.Trial{x}.States, 'Reward') && ...
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
            elseif ( isfield(Data.RawEvents.Trial{x}.States, 'EarlyPress1Punish') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.EarlyPress1Punish(1)))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp('Outcome: EarlyPress1Punish');
                end
                Outcomes(x) = 0;    % draws red circle on outcome plot
            elseif ( isfield(Data.RawEvents.Trial{x}.States, 'EarlyPress2Punish') && ...
                    ~isnan(Data.RawEvents.Trial{x}.States.EarlyPress2Punish(1)))
                if (x == Data.nTrials && isEndOfTrial)  % only print outcome to console for the trial that just occured
                    disp('Outcome: EarlyPress2Punish');
                end
                Outcomes(x) = 0;    % draws red circle on outcome plot                
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
        TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update', Data.nTrials+1, TrialTypes, OptoTrialTypes, ProbeTrialTypes, Outcomes);
    else
        Outcomes = zeros(1, 1);
        TrialTypeOutcomePlot(BpodSystem.GUIHandles.OutcomePlot,'update', 1, TrialTypes, OptoTrialTypes, ProbeTrialTypes, Outcomes);
        yticks([1 2])
        yticklabels({'left','right'})
    end
end


    end
end