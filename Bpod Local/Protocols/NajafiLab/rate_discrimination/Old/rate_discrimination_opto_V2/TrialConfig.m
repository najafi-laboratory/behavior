classdef TrialConfig
    methods


%% trial generation

function [TrialTypes] = GenTrials(obj, S)
    TrialTypes = ceil(rand(1, 1106)*2); 
end


function [GrayPerturbISI] = SetPostPertISI( ...
        obj, S, TrialTypes, currentTrial, PostPertISI)
    switch TrialTypes(currentTrial)
        case 1 % trial is left with short ISI
            GrayPerturbISI = S.GUI.ISIOrig_s - PostPertISI;
        case 2 % trial is right with long ISI
            GrayPerturbISI = S.GUI.ISIOrig_s + PostPertISI;
    end
end


%% random isi


function [JitterFlag] = GetJitterFlag( ...
        obj, S)
    if (S.GUI.ActRandomISI == 1)
        JitterFlag = 1;
    else
        JitterFlag = 0;
    end
end


%% change trial difficulty


function [TrialDifficulty] = SamplingDiff( ...
        obj, S)
    % 1 - Easy, 2 - MediumEasy, 3 - MediumHard, 4 - Hard
    DifficultyLevels = [1, 2, 3, 4];
    sum = S.GUI.PercentEasy + S.GUI.PercentMediumEasy + S.GUI.PercentMediumHard + S.GUI.PercentHard;
    FractionEasy = S.GUI.PercentEasy / sum;
    FractionMediumEasy = S.GUI.PercentMediumEasy / sum;
    FractionMediumHard = S.GUI.PercentMediumHard / sum;
    FractionHard = S.GUI.PercentHard / sum;
    DifficultyProbabilities = [FractionEasy, FractionMediumEasy, FractionMediumHard, FractionHard];
    CummuProb = [0, cumsum(DifficultyProbabilities)];
    r = rand;
    ind = find(r>CummuProb, 1, 'last');
    TrialDifficulty = DifficultyLevels(ind);
end




%% anti-bias algorithm

% get valve time from valve amount
function [ValveTime] = Amount2Time( ...
        obj, Amount, Valve)
    ValveTime = GetValveTimes(Amount, Valve);
    if (ValveTime < 0)
        ValveTime = 0;
    end
end


% automatically adjust side valve
function [AntiBiasVar, LeftValveAmount_uL, RightValveAmount_uL, TrialTypes] = AntiBiasValveAdjust( ...
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
        BpodSystem, AntiBiasVar, currentTrial, TrialTypes)
        if (currentTrial > 1)
            if (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'PunishNaive') && ...
                ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.PunishNaive(1)))
                [AntiBiasVar] = AddLastCorrectness(AntiBiasVar, TrialTypes, currentTrial, 0);
            elseif (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'RewardNaive') && ...
                    ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.RewardNaive(1)))
                [AntiBiasVar] = AddLastCorrectness(AntiBiasVar, TrialTypes, currentTrial, 1);
            elseif (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'Punish') && ...
                    ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Punish(1)))
                [AntiBiasVar] = AddLastCorrectness(AntiBiasVar, TrialTypes, currentTrial, 0);
            elseif (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'Reward') && ...
                    ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1)))
                [AntiBiasVar] = AddLastCorrectness(AntiBiasVar, TrialTypes, currentTrial, 1);
            elseif (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'ChangingMindReward') && ...
                    ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.ChangingMindReward(1)))
                [AntiBiasVar] = AddLastCorrectness(AntiBiasVar, TrialTypes, currentTrial, 0);
            else
                AntiBiasVar.CompletedHist = AntiBiasVar.CompletedHist;
            end
        end
    end
    % compute bias index for bias detection
    function [AntiBiasVar] = GetBiasIndex( ...
            S, AntiBiasVar)
        LeftACC  = sum(AntiBiasVar.CompletedHist.left(end-S.GUI.NumMonitorTrials+1:end)) / S.GUI.NumMonitorTrials;
        RightACC = sum(AntiBiasVar.CompletedHist.right(end-S.GUI.NumMonitorTrials+1:end)) / S.GUI.NumMonitorTrials;
        AntiBiasVar.BiasIndex = LeftACC - RightACC;
    end
    % update whether adjustment is necessary
    function [AntiBiasVar] = BiasDetection( ...
            S, AntiBiasVar)
        if (length(AntiBiasVar.CompletedHist.left) > S.GUI.NumMonitorTrials && ...
            length(AntiBiasVar.CompletedHist.right) > S.GUI.NumMonitorTrials)
            [AntiBiasVar] = GetBiasIndex(S, AntiBiasVar);
            % left bias
            if (AntiBiasVar.BiasIndex >= S.GUI.BiasIndexThres)
                AntiBiasVar.ValveFlag = 'LeftBias';
            % right bias
            elseif (AntiBiasVar.BiasIndex <= - S.GUI.BiasIndexThres)
                AntiBiasVar.ValveFlag = 'RightBias';
            % no bias
            elseif (AntiBiasVar.BiasIndex > - S.GUI.BiasIndexThres && ...
                    AntiBiasVar.BiasIndex <   S.GUI.BiasIndexThres)
                AntiBiasVar.ValveFlag = 'NoBias';
            % keep
            else
                AntiBiasVar.ValveFlag = AntiBiasVar.ValveFlag;
            end
        end
    end
    % adjust valve time according to bias flag
    function [LeftValveAmount_uL, RightValveAmount_uL] = UpdateValveTime( ...
            S, AntiBiasVar)
        switch AntiBiasVar.ValveFlag
            case 'NoBias' % no bias
                LeftValveAmount_uL = S.GUI.LeftValveAmount_uL;
                RightValveAmount_uL = S.GUI.RightValveAmount_uL;
            case 'LeftBias' % left bias
                LeftValveAmount_uL = S.GUI.LeftValveAmount_uL * S.GUI.AdjustValvePercent;
                RightValveAmount_uL = S.GUI.RightValveAmount_uL;
            case 'RightBias' % right bias
                LeftValveAmount_uL = S.GUI.LeftValveAmount_uL;
                RightValveAmount_uL = S.GUI.RightValveAmount_uL * S.GUI.AdjustValvePercent;

        end
    end
    % update trial type with fraction according to bias flag with
    % AdjustFraction>0.5
    function [TrialTypes] = UpdateTrialType( ...
            S, AntiBiasVar, TrialTypes, currentTrial)
        switch AntiBiasVar.ValveFlag
            case 'NoBias' % no bias
                TrialTypes = TrialTypes;
            case 'LeftBias' % left bias
                SideProbabilities = [1-S.GUI.AdjustFraction, S.GUI.AdjustFraction];
                [TrialSide] = SampleSide(obj, SideProbabilities);
                TrialTypes(currentTrial) = TrialSide;
            case 'RightBias' % right bias
                SideProbabilities = [S.GUI.AdjustFraction, 1-S.GUI.AdjustFraction];
                [TrialSide] = SampleSide(obj, SideProbabilities);
                TrialTypes(currentTrial) = TrialSide;
        end
    end
    % main process
    if (S.GUI.AdjustValve == 1)
        [AntiBiasVar] = UpdateCompletedHist( ...
            BpodSystem, AntiBiasVar, currentTrial, TrialTypes);
        [AntiBiasVar] = BiasDetection( ...
                S, AntiBiasVar);
        [LeftValveAmount_uL, RightValveAmount_uL] = UpdateValveTime( ...
                S, AntiBiasVar);
        [TrialTypes] = UpdateTrialType( ...
                S, AntiBiasVar, TrialTypes, currentTrial);
    end
end


% repeat incorrect trials until it is correct 
function [TrialTypes, AntiBiasVar] = RepeatedIncorrect( ...
        obj, BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes)
    if (currentTrial > 1)
        if (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'PunishNaive') && ...
            ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.PunishNaive(1)))
            AntiBiasVar.IncorrectFlag = 1;
            AntiBiasVar.IncorrectType = TrialTypes(currentTrial-1);
        elseif (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'RewardNaive') && ...
                ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.RewardNaive(1)))
            AntiBiasVar.IncorrectFlag = 0;
        elseif (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'Punish') && ...
                ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Punish(1)))
            AntiBiasVar.IncorrectFlag = 1;
            AntiBiasVar.IncorrectType = TrialTypes(currentTrial-1);
        elseif (isfield(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States, 'Reward') && ...
                ~isnan(BpodSystem.Data.RawEvents.Trial{currentTrial-1}.States.Reward(1)))
            AntiBiasVar.IncorrectFlag = 0;
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
        [TrialSide] = SampleSide(obj, SideProbabilities);
        TrialTypes(currentTrial) = TrialSide;
    else
        TrialTypes = TrialTypes;
    end
end


%% trial adjustment


% sample a trial type from given side probability
function [TrialSide] = SampleSide( ...
        obj, prob)
    Sides = [1, 2];
    cp = [0, cumsum(prob)]; % cumulative probability -> use as interval to pick Left or Right
    r = rand; % get random scalar drawn from the uniform distribution in the interval (0,1).
    ind = find(r>cp, 1, 'last');  % get discrete index (1 or 2 for left or right)
    TrialSide = Sides(ind);
end


% reduce consecutive trials
function [TrialTypes] = AdjustWarmupTrials( ...
        obj, S, TrialTypes)       
    MaxSameConsecutiveTrials = 7;
    for i = MaxSameConsecutiveTrials+1:S.GUI.MaxTrials
        PrevMaxTrials = TrialTypes(i-MaxSameConsecutiveTrials:i-1);
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


%% easymax


function [PerturbDurMin, PerturbDurMax] = GetPerturDurMinMax( ...
        obj, TrialDifficulty, PerturbInterval, PerturbDurFullRange)
    switch TrialDifficulty
        case 1
            PerturbDurMin = PerturbInterval.EasyMinPercent*PerturbDurFullRange;
            PerturbDurMax = PerturbInterval.EasyMaxPercent*PerturbDurFullRange;
        case 2
            PerturbDurMin = PerturbInterval.MediumEasyMinPercent*PerturbDurFullRange;
            PerturbDurMax = PerturbInterval.MediumEasyMaxPercent*PerturbDurFullRange;
        case 3
            PerturbDurMin = PerturbInterval.MediumHardMinPercent*PerturbDurFullRange;
            PerturbDurMax = PerturbInterval.MediumHardMaxPercent*PerturbDurFullRange;            
        case 4
            PerturbDurMin = PerturbInterval.HardMinPercent*PerturbDurFullRange;
            PerturbDurMax = PerturbInterval.HardMaxPercent*PerturbDurFullRange;
    end
end


% default: naive/mid1 is activated. mid2/well is deactivated.
function [PostPertISI, EasyMaxInfo] = GetPostPertISI( ...
        obj, S, TrialDifficulty, PerturbInterval)
    PerturbDurFullRange = S.GUI.ISIOrig_s * 1000 - S.GUI.MinISIPerturb_ms;
    [PerturbDurMin, PerturbDurMax] = GetPerturDurMinMax( ...
        obj, TrialDifficulty, PerturbInterval, PerturbDurFullRange);
    switch S.GUI.EasyMax
        case 1 % default
            if (S.GUI.TrainingLevel == 1 || S.GUI.TrainingLevel == 2 || S.GUI.TrainingLevel == 3)
                PostPertISI = PerturbDurMax/1000;
                EasyMaxInfo = 'Activated';
            else
                PostPertISI = unifrnd(PerturbDurMin, PerturbDurMax, 1, 1)/1000; % get random duration from calculated range, convert to seconds
                EasyMaxInfo = 'Deactivated';
            end
        case 2 % manually activated
            PostPertISI = PerturbDurMax/1000;
            EasyMaxInfo = 'Activated';
        case 3 % manually deactivated
            PostPertISI = unifrnd(PerturbDurMin, PerturbDurMax, 1, 1)/1000;
            EasyMaxInfo = 'Deactivated';
    end
end


%% ITI and timeout ITI


function [ITI] = GetITI( ...
        obj, S)
    ITI = 19961106;
    if (S.GUI.SetManualITI == 1)
        ITI = S.GUI.ManualITI;
    else
        while (ITI < S.GUI.ITIMin || ITI > S.GUI.ITIMax)
            ITI = -log(rand) * S.GUI.ITIMean;
        end
    end
end


function [TimeOutPunish] = GetTimeOutPunish( ...
        obj, S)
    if (S.GUI.ActTimeOutPunish == 1)
        if (S.GUI.ManuallTimeOutPunish)
            TimeOutPunish = S.GUI.TimeOutPunish;
        else
            switch S.GUI.TrainingLevel
                case 1
                    TimeOutPunish = 1.5;
                case 2
                    TimeOutPunish = 2.0;
                case 3
                    TimeOutPunish = 2.5;
                case 4
                    TimeOutPunish = 3.0;
            end
        end
    else
        TimeOutPunish = 0;
    end
end


%% choice window

function [ChoiceWindow] = GetChoiceWindow( ...
        obj, S)
    if (S.GUI.ManualChoiceWindow)
        ChoiceWindow = S.GUI.ChoiceWindow_s;
    else
        switch S.GUI.TrainingLevel
            case 1
                ChoiceWindow = 15;
            case 2
                ChoiceWindow = 10;
            case 3
                ChoiceWindow = 5;
            case 4
                ChoiceWindow = 5;
        end
    end
end


function [ChangeMindDur] = GetChangeMindDur( ...
        obj, S)
    if (S.GUI.ManuallChangeMindDur)
        ChangeMindDur = S.GUI.ChangeMindDur;
    else
        switch S.GUI.TrainingLevel
            case 1
                ChangeMindDur = 10;
            case 2
                ChangeMindDur = 5;
            case 3
                ChangeMindDur = 0;
            case 4
                ChangeMindDur = 0;
        end
    end
end



%% moving spouts


function SetMotorPos = ConvertMaestroPos(obj, MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end

































    end
end