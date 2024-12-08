classdef TrialConfig
    methods


%% trial generation

function [TrialTypes] = GenTrials(obj, S)
    TrialTypes = ceil(rand(1, 1106)*2); 
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
        disp(['Bias Index: ', num2str(AntiBiasVar.BiasIndex)]);
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
                disp('Left Bias Detected');
            % right bias
            elseif (AntiBiasVar.BiasIndex <= - S.GUI.BiasIndexThres)
                AntiBiasVar.ValveFlag = 'RightBias';
                disp('Right Bias Detected');
            % no bias
            elseif (AntiBiasVar.BiasIndex > - S.GUI.BiasIndexThres && ...
                    AntiBiasVar.BiasIndex <   S.GUI.BiasIndexThres)
                AntiBiasVar.ValveFlag = 'NoBias';
                disp('No Bias');
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
                disp('No Valve Adjust');
            case 'LeftBias' % left bias
                LeftValveAmount_uL = S.GUI.LeftValveAmount_uL * S.GUI.AdjustValvePercent;
                RightValveAmount_uL = S.GUI.RightValveAmount_uL;
                disp('Reduce Left Valve 25%');
            case 'RightBias' % right bias
                LeftValveAmount_uL = S.GUI.LeftValveAmount_uL;
                RightValveAmount_uL = S.GUI.RightValveAmount_uL * S.GUI.AdjustValvePercent;
                disp('Reduce Right Valve 25%');
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
                disp('Reduce Left trial to 10%');
                disp('Increase Right trial to 90%');
            case 'RightBias' % right bias
                SideProbabilities = [S.GUI.AdjustFraction, 1-S.GUI.AdjustFraction];
                [TrialSide] = SampleSide(obj, SideProbabilities);
                TrialTypes(currentTrial) = TrialSide;
                disp('Reduce Right trial to 10%');
                disp('Increase Left trial to 90%');                
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

% automatically adjust servo
function [AntiBiasVar] = AntiBiasServoAdjust( ...
        obj, BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes)
    % global AntiBiasVar
    switch AntiBiasVar.ValveFlag
        case 'NoBias' % no bias
            AntiBiasVar = AntiBiasVar;
        case 'LeftBias' % left bias
            if abs(AntiBiasVar.ServoRightAdjust) < S.GUI.ServoIncrementMax
                if AntiBiasVar.ServoRightTrialsSinceAdjust >= 3
                    AntiBiasVar.ServoRightTrialsSinceAdjust = 0;
                    AntiBiasVar.ServoRightAdjust = AntiBiasVar.ServoRightAdjust - S.GUI.ServoBiasIncrement;
                    disp('Setting Right Spout Closer');    
                else
                    AntiBiasVar.ServoRightTrialsSinceAdjust = AntiBiasVar.ServoRightTrialsSinceAdjust + 1;
                end
            end
        case 'RightBias' % right bias
            if abs(AntiBiasVar.ServoLeftAdjust) < S.GUI.ServoIncrementMax
                if AntiBiasVar.ServoLeftTrialsSinceAdjust >= 3
                    AntiBiasVar.ServoLeftTrialsSinceAdjust = 0;
                    AntiBiasVar.ServoLeftAdjust = AntiBiasVar.ServoLeftAdjust + S.GUI.ServoBiasIncrement;
                    disp('Setting Left Spout Closer'); 
                else
                    AntiBiasVar.ServoLeftTrialsSinceAdjust = AntiBiasVar.ServoLeftTrialsSinceAdjust + 1;
                end
            end        
    end
    disp(['ServoRightAdjust: ', num2str(AntiBiasVar.ServoRightAdjust)]);    
    disp(['ServoRightTrialsSinceAdjust: ', num2str(AntiBiasVar.ServoRightTrialsSinceAdjust)]);
    disp(['ServoLeftAdjust: ', num2str(AntiBiasVar.ServoLeftAdjust)]);  
    disp(['ServoLeftTrialsSinceAdjust: ', num2str(AntiBiasVar.ServoLeftTrialsSinceAdjust)]);
end

% repeat incorrect trials until it is correct 
function [TrialTypes, AntiBiasVar] = RepeatedIncorrect( ...
        obj, BpodSystem, S, AntiBiasVar, currentTrial, TrialTypes)
    if (currentTrial > 1)
        % if previous trial was punish, set incorrectflag to 1, otherwise
        % set incorrectflag to 0
        % if incorrectflag set to 1, get trial type (left or right)
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
        % if repeatedincorrect anti-bias enabled, and previous trial was
        % punish, then draw probabilities for current trial based on trial
        % type (left or right)
        if (S.GUI.RepeatedIncorrect == 1)
            disp('Previous Trial Incorrect.  Adjusting repeated probability.')
            if (AntiBiasVar.IncorrectFlag == 1)
                switch AntiBiasVar.IncorrectType
                    case 1
                        SideProbabilities = [S.GUI.RepeatedProb, 1-S.GUI.RepeatedProb];
                        disp('Previous Left Trial Incorrect.  Update to Left Side Trial.')
                    case 2
                        SideProbabilities = [1-S.GUI.RepeatedProb, S.GUI.RepeatedProb];
                        disp('Previous Right Trial Incorrect.  Update to Right Side Trial.')
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
        obj, S, TrialDifficulty, PerturbInterval, TrialTypes, currentTrial)
    % PerturbDurFullRange = S.GUI.ISIOrig_s * 1000 - S.GUI.MinISIPerturb_ms;
    PerturbDurFullRange = S.GUI.ISIShortMax_s;
    [PerturbDurMin, PerturbDurMax] = GetPerturDurMinMax( ...
        obj, TrialDifficulty, PerturbInterval, PerturbDurFullRange);
    switch S.GUI.EasyMax
        case 1 % default
            if (S.GUI.TrainingLevel == 1 || S.GUI.TrainingLevel == 2 || S.GUI.TrainingLevel == 3 || S.GUI.TrainingLevel == 4)
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

    % for now, skip difficulty calculation
    EasyMaxInfo = 'Deactivated';
    % PostPertISI = 0;
    switch S.GUI.TrainingLevel
        case 1 % Naive
            PostPertISI = GetISIMean(obj, S, TrialTypes, currentTrial);
        case 2 % Early       
            PostPertISI = GetISIMean(obj, S, TrialTypes, currentTrial);
        case 3 % Mid1       
            PostPertISI = GetISIFromDist(obj, S, TrialTypes, currentTrial);
        case 4 % Mid2
            PostPertISI = GetISIFromDist(obj, S, TrialTypes, currentTrial);
        case 5 % Well
            PostPertISI = GetISIFromDist(obj, S, TrialTypes, currentTrial);
    end   
end

function [PostPertISI] = GetISIMean(obj, S, TrialTypes, currentTrial)
    switch TrialTypes(currentTrial)
        case 1 % trial is left with short ISI
            PostPertISI = S.GUI.ISIShortMean_s;
        case 2 % trial is right with long ISI
            PostPertISI = S.GUI.ISILongMean_s;
    end     
end

function [PostPertISI] = GetISIFromDist(obj, S, TrialTypes, currentTrial)
    switch TrialTypes(currentTrial)
        case 1 % trial is left with short ISI
            PostPertISI = DrawFromUniform(obj, S.GUI.ISIShortMin_s, S.GUI.ISIShortMax_s)
        case 2 % trial is right with long ISI
            PostPertISI = DrawFromUniform(obj, S.GUI.ISILongMin_s, S.GUI.ISILongMax_s);
    end 
end

function [GrayPerturbISI] = SetPostPertISI( ...
        obj, S, TrialTypes, currentTrial, PostPertISI)
    % switch TrialTypes(currentTrial)
    %     case 1 % trial is left with short ISI
    %         GrayPerturbISI = S.GUI.ISIOrig_s - PostPertISI;
    %     case 2 % trial is right with long ISI
    %         GrayPerturbISI = S.GUI.ISIOrig_s + PostPertISI;
    % end
    GrayPerturbISI = PostPertISI;
end

%% ITI and timeout ITI


function [ITI] = GetITI( ...
        obj, S)
    ITI = 19961106;
    if (S.GUI.SetManualITI == 1)
        ITI = S.GUI.ManualITI;
    else
        % exponential distribution
        % while (ITI < S.GUI.ITIMin || ITI > S.GUI.ITIMax)
        %     ITI = -log(rand) * S.GUI.ITIMean;
        % end
        
        % uniform distribution
        ITI = DrawFromUniform(obj, S.GUI.ITIMin, S.GUI.ITIMax);

        switch S.GUI.TrainingLevel
            case 1                
                ITI = S.GUI.ITIMin;
            case 2
                ITI = S.GUI.ITIMin;                
            case 3
                ITI = S.GUI.ITIMean;
            case 4
                ITI = 2.5;
            case 5
                ITI = 3.0;
        end        
    end
end

function [r] = DrawFromUniform(obj, lb, ub)
    % uniform distribution
    a = lb;  % Lower bound
    b = ub;  % Upper bound    
    r = a + (b-a)*rand();  % Single random value between a and b    
end

function [TimeOutPunish] = GetTimeOutPunish( ...
        obj, S)
    if (S.GUI.ActTimeOutPunish == 1)
        if (S.GUI.ManuallTimeOutPunish)
            TimeOutPunish = S.GUI.TimeOutPunish;
        else
            switch S.GUI.TrainingLevel
                case 1 % Naive
                    TimeOutPunish = 2.0;
                case 2 % Early
                    TimeOutPunish = 2.0;                    
                case 3 % Mid1
                    TimeOutPunish = 2.0;
                case 4 % Mid2
                    TimeOutPunish = 2.0;
                case 5 % Well
                    TimeOutPunish = 2.0;
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
                ChoiceWindow = 10;
            case 2
                ChoiceWindow = 5;                
            case 3
                ChoiceWindow = 5;
            case 4
                ChoiceWindow = 3;
            case 5
                ChoiceWindow = 1;
        end
    end
end


function [ChangeMindDur] = GetChangeMindDur( ...
        obj, S)
    if (S.GUI.ManuallChangeMindDur)
        ChangeMindDur = S.GUI.ChangeMindDur;
    else
        switch S.GUI.TrainingLevel
            case 1 % naive
                ChangeMindDur = 10;
            case 2 % early
                ChangeMindDur = 5;                
            case 3 % mid 1
                ChangeMindDur = 5;
            case 4 % mid 2
                ChangeMindDur = 0;
            case 5 % well
                ChangeMindDur = 0;
        end
    end
end

function [PostVisStimDelay] = GetPostVisStimDelay( ...
        obj, S)
    switch S.GUI.TrainingLevel
        case 1 % naive
            PostVisStimDelay = S.GUI.PostVisStimDelayMin_s;
        case 2 % early
            PostVisStimDelay = DrawFromUniform(obj, S.GUI.PostVisStimDelayMin_s, S.GUI.PostVisStimDelayMax_s);
        case 3 % mid 1
            PostVisStimDelay = S.GUI.PostVisStimDelayMean_s;
        case 4 % mid 2
            PostVisStimDelay = S.GUI.PostVisStimDelayMin_s;
        case 5 % well
            PostVisStimDelay = S.GUI.PostVisStimDelayMin_s;
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