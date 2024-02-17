classdef TrialConfig
    methods


%% anti-bias algorithm


% automatically adjust side valve
function [AntiBiasVar, LeftValveTime, RightValveTime, TrialTypes] = AntiBiasValveAdjust( ...
        obj, BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes)
    % add the last outcome to history
    function [AntiBiasVar] = AddLastCorrectness( ...
            AntiBiasVar, TrialTypes, currentTrial, correctness)
        switch TrialTypes(currentTrial-1)
            case 1
                AntiBiasVar.CompletedHist.left(end+1) = correctness;
            case 2
                AntiBiasVar.CompletedHist.right(end+1) = correctness;
        end
    end
    % update history for different training levels
    function [AntiBiasVar] = UpdateCompletedHist( ...
        BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes)
        if (currentTrial > 1)
            if (S.GUI.TrainingLevel == 1) % Habituation
            elseif (S.GUI.TrainingLevel == 2) % Naive
                if (~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.PunishNaive(1)))
                    [AntiBiasVar] = AddLastCorrectness(AntiBiasVar, TrialTypes, currentTrial, -1);
                elseif (~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.RewardNaive(1)))
                    [AntiBiasVar] = AddLastCorrectness(AntiBiasVar, TrialTypes, currentTrial, 1);
                else
                    AntiBiasVar.CompletedHist = AntiBiasVar.CompletedHist;
                end
            else % mid1/mid2/well
                if (~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Punish(1)))
                    [AntiBiasVar] = AddLastCorrectness(AntiBiasVar, TrialTypes, currentTrial, -1);
                elseif (~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1)))
                    [AntiBiasVar] = AddLastCorrectness(AntiBiasVar, TrialTypes, currentTrial, 1);
                else
                    AntiBiasVar.CompletedHist = AntiBiasVar.CompletedHist;
                end
            end
        end
    end
    % update whether adjustment is necessary
    function [AntiBiasVar] = BiasDetection( ...
            S, AntiBiasVar)
        if (length(AntiBiasVar.CompletedHist.left) > S.GUI.AdjustValveThres && ...
            length(AntiBiasVar.CompletedHist.right) > S.GUI.AdjustValveThres)
            % left bias
            if (sum(AntiBiasVar.CompletedHist.left(end-S.GUI.AdjustValveThres+1:end)) >= S.GUI.AdjustValveThres && ...
                sum(AntiBiasVar.CompletedHist.right(end-S.GUI.AdjustValveThres+1:end)) <= -S.GUI.AdjustValveThres)
                AntiBiasVar.ValveFlag = -1;
            % right bias
            elseif (sum(AntiBiasVar.CompletedHist.right(end-S.GUI.AdjustValveThres+1:end)) >= S.GUI.AdjustValveThres && ...
                    sum(AntiBiasVar.CompletedHist.left(end-S.GUI.AdjustValveThres+1:end)) <= -S.GUI.AdjustValveThres)
                AntiBiasVar.ValveFlag = 1;
            % no bias
            elseif (sum(AntiBiasVar.CompletedHist.right(end-S.GUI.AdjustValveThres+1:end)) >= S.GUI.AdjustValveThres && ...
                    sum(AntiBiasVar.CompletedHist.left(end-S.GUI.AdjustValveThres+1:end)) >= S.GUI.AdjustValveThres)
                AntiBiasVar.ValveFlag = 0;
            % keep
            else
                AntiBiasVar.ValveFlag = AntiBiasVar.ValveFlag;
            end
        end
    end
    % adjust valve time according to bias flag
    function [LeftValveTime, RightValveTime] = UpdateValveTime( ...
            S, AntiBiasVar)
        switch AntiBiasVar.ValveFlag
            case 0 % no bias
                LeftValveTime = S.GUI.LeftValveTime_s;
                RightValveTime = S.GUI.RightValveTime_s;
            case -1 % left bias
                LeftValveTime = S.GUI.AdjustValveTime;
                RightValveTime = S.GUI.RightValveTime_s;
            case 1 % right bias
                LeftValveTime = S.GUI.LeftValveTime_s;
                RightValveTime = S.GUI.AdjustValveTime;

        end
    end
    % update trial type with fraction according to bias flag with
    % AdjustFraction>0.5
    function [TrialTypes] = UpdateTrialType( ...
            S, AntiBiasVar, TrialTypes, currentTrial)
        switch AntiBiasVar.ValveFlag
            case 0 % no bias
                TrialTypes = TrialTypes;
            case -1 % left bias need more right trials
                SideProbabilities = [1-S.GUI.AdjustFraction, S.GUI.AdjustFraction];
                [TrialSide] = SampleSide(obj, SideProbabilities);
                TrialTypes(currentTrial) = TrialSide;
            case 1 % right bias need more left trials
                SideProbabilities = [S.GUI.AdjustFraction, 1-S.GUI.AdjustFraction];
                [TrialSide] = SampleSide(obj, SideProbabilities);
                TrialTypes(currentTrial) = TrialSide;
        end
    end
    % main process
    if (S.GUI.AdjustValve == 1)
        [AntiBiasVar] = UpdateCompletedHist( ...
            BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes);
        [AntiBiasVar] = BiasDetection( ...
                S, AntiBiasVar);
        [LeftValveTime, RightValveTime] = UpdateValveTime( ...
                S, AntiBiasVar);
        [TrialTypes] = UpdateTrialType( ...
                S, AntiBiasVar, TrialTypes, currentTrial);
    end
end


% repeat incorrect trials until it is correct 
function [TrialTypes, AntiBiasVar] = RepeatedIncorrect( ...
        obj, BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes)
    if (currentTrial > 1)
        if (S.GUI.TrainingLevel == 1) % Habituation
        elseif (S.GUI.TrainingLevel == 2) % Naive
            if (~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.PunishNaive(1)))
                AntiBiasVar.IncorrectFlag = 1;
                AntiBiasVar.IncorrectType = TrialTypes(currentTrial-1);
            elseif (~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.RewardNaive(1)))
                AntiBiasVar.IncorrectFlag = 0;
            end
        else % mid1/mid2/well
            if (~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Punish(1)))
                AntiBiasVar.IncorrectFlag = 1;
                AntiBiasVar.IncorrectType = TrialTypes(currentTrial-1);
            elseif (~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1)))
                AntiBiasVar.IncorrectFlag = 0;
            end
        end
        if (S.GUI.RepeatedIncorrect == 1)
            if (AntiBiasVar.IncorrectFlag == 1)
                switch AntiBiasVar.IncorrectType
                    case 1
                        SideProbabilities = [S.GUI.RepeatedProb, 1-S.GUI.RepeatedProb];
                    case 2
                        SideProbabilities = [1-S.GUI.RepeatedProb, S.GUI.RepeatedProb];
                end
                [TrialSide] = SampleSide(obj, SideProbabilities);
                TrialTypes(currentTrial) = TrialSide;
            end
        end
    end
end


% manuall trial fraction setting
function [TrialTypes] = ManuallFraction( ...
        obj, S, currentTrial, TrialTypes)
    if (S.GUI.ShortISIFraction ~= 0.5)
        SideProbabilities = [S.GUI.ShortISIFraction, 1-S.GUI.ShortISIFraction];
        [TrialSide] = SampleSide(obj, SideProbabilities)
        TrialTypes(currentTrial) = TrialSide;
    else
        TrialTypes = TrialTypes;
    end
end


%%

% sample a trial type from given side probability
function [TrialSide] = SampleSide( ...
        obj, prob)
    Sides = [1, 2];
    cp = [0, cumsum(prob)]; % cumulative probability -> use as interval to pick Left or Right
    r = rand; % get random scalar drawn from the uniform distribution in the interval (0,1).
    ind = find(r>cp, 1, 'last');  % get discrete index (1 or 2 for left or right)
    TrialSide = Sides(ind);
end


% reduce consecutive trials in warmup
function [TrialTypes] = AdjustWarmupTrials( ...
        obj, S, TrialTypes)       
    MaxSameConsecutiveTrials = 4;
    for i = MaxSameConsecutiveTrials:S.GUI.NumEasyWarmupTrials 
        if (i > MaxSameConsecutiveTrials)
            PrevMaxTrials = TrialTypes(i-3:i-1);
            if (all(PrevMaxTrials == 1) || all(PrevMaxTrials == 2))
                NewSameAsPrevMax = true;
                while NewSameAsPrevMax
                    DrawTrialType = unidrnd(2,1,1);       
                    if ~all(PrevMaxTrials == DrawTrialType)
                        NewSameAsPrevMax = false;
                    end
                end
                TrialTypes(i) = DrawTrialType;
            end
        end   
    end
end


function [RandomPerturbationDur, EasyMaxInfo] = GetPerturbationDur( ...
        obj, S, EasyMax, PerturbDurMin, PerturbDurMax)
    switch S.GUI.EasyMax
        case 1 % default
            if (S.GUI.TrainingLevel == 1 | S.GUI.TrainingLevel == 2 | S.GUI.TrainingLevel == 3)
                RandomPerturbationDur = EasyMax;
                EasyMaxInfo = 'Activated';
            else
                RandomPerturbationDur = unifrnd(PerturbDurMin, PerturbDurMax, 1, 1)/1000; % get random duration from calculated range, convert to seconds
                EasyMaxInfo = 'Deactivated';
            end
        case 2 % manually activated
            RandomPerturbationDur = EasyMax;
            EasyMaxInfo = 'Activated';
        case 3 % manually deactivated
            RandomPerturbationDur = unifrnd(PerturbDurMin, PerturbDurMax, 1, 1)/1000;
            EasyMaxInfo = 'Deactivated';
    end
end


function [wait_dur] = GetWaitDur( ...
        obj, BpodSystem, S, wait_dur, currentTrial, LastNumEasyWarmupTrials, VisStimDuration)
    if currentTrial < LastNumEasyWarmupTrials+1
        wait_dur = 0;
    elseif currentTrial == LastNumEasyWarmupTrials+1
        wait_dur = S.GUI.WaitDurOrig_s;
    else
        if currentTrial>1 && ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.CenterReward(1))
            wait_dur = min(wait_dur + S.GUI.WaitDurStep_s, VisStimDuration);
        end
    end
end


function [ITI] = GetITI( ...
        obj, S)
    ITI = 19840124;
    while (ITI < S.GUI.ITIMin || ITI > S.GUI.ITIMax)
        ITI = -log(rand) * S.GUI.ITIMean;
    end
    if (S.GUI.ForceITIZero)
        ITI = 0;
    end
end


function [TimeOutPunish] = GetTimeOutPunish( ...
        obj, S)
    if (S.GUI.ActTimeOutPunish)
        TimeOutPunish = 19961106;
        if (S.GUI.ManuallTimeOutPunish)
            TimeOutPunishMean = S.GUI.TimeOutPunishMean;
            while TimeOutPunish < S.GUI.TimeOutPunishMin || TimeOutPunish > S.GUI.TimeOutPunishMax
                TimeOutPunish = -log(rand) * TimeOutPunishMean;
            end
        else
            switch S.GUI.TrainingLevel
                case 1
                    TimeOutPunish = 0;
                case 2
                    TimeOutPunish = 0;
                case 3
                    TimeOutPunishMean = 1;
                    while TimeOutPunish < S.GUI.TimeOutPunishMin || TimeOutPunish > S.GUI.TimeOutPunishMax
                        TimeOutPunish = -log(rand) * TimeOutPunishMean;
                    end
                case 4
                    TimeOutPunishMean = 2;
                    while TimeOutPunish < S.GUI.TimeOutPunishMin || TimeOutPunish > S.GUI.TimeOutPunishMax
                        TimeOutPunish = -log(rand) * TimeOutPunishMean;
                    end
                case 5
                    TimeOutPunishMean = 3;
                    while TimeOutPunish < S.GUI.TimeOutPunishMin || TimeOutPunish > S.GUI.TimeOutPunishMax
                        TimeOutPunish = -log(rand) * TimeOutPunishMean;
                    end
            end
        end
    else
        TimeOutPunish = 0;
    end
end




































    end
end