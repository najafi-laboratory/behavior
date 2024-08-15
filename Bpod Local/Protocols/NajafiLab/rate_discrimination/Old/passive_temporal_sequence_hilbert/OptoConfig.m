classdef OptoConfig
    methods

function [OptoTrialTypes] = GenOptoTrials(obj, BpodSystem, S)
    BpodSystem.Data.PreviousSessionType = S.GUI.SessionType;
    BpodSystem.Data.PreviousOptoTrialTypeSeq = S.GUI.OptoTrialTypeSeq;
    BpodSystem.Data.PreviousOnFraction = S.GUI.OnFraction;
    BpodSystem.Data.PreviousNumOptoTrialsPerBlock = S.GUI.NumOptoTrialsPerBlock;
    OptoTrialTypes = ceil(rand(1, S.GUI.MaxImg)*2);
end

function [OptoTrialTypes] = UpdateOptoTrials(obj, BpodSystem, S, OptoTrialTypes, currentTrial, forceUpdate)
    updateOptoTrialTypeSequence = 0;

    if forceUpdate == 1
        updateOptoTrialTypeSequence = 1;
    end

    if S.GUI.SessionType == 2
            S.GUI.OptoTrialTypeSeq = 1;
            S.GUI.OnFraction = 0;
            updateOptoTrialTypeSequence = 1;
    end

    if (BpodSystem.Data.PreviousSessionType ~= S.GUI.SessionType) && (S.GUI.SessionType == 1)
        updateOptoTrialTypeSequence = 1;           
    end
    BpodSystem.Data.PreviousSessionType = S.GUI.SessionType;

    if BpodSystem.Data.PreviousOptoTrialTypeSeq ~= S.GUI.OptoTrialTypeSeq
        updateOptoTrialTypeSequence = 1;
    end
    BpodSystem.Data.PreviousOptoTrialTypeSeq = S.GUI.OptoTrialTypeSeq;

    if BpodSystem.Data.PreviousNumOptoTrialsPerBlock ~= S.GUI.NumOptoTrialsPerBlock
        updateOptoTrialTypeSequence = 1;
        BpodSystem.Data.PreviousNumOptoTrialsPerBlock = S.GUI.NumOptoTrialsPerBlock;
    end

    if BpodSystem.Data.PreviousOnFraction ~= S.GUI.OnFraction
        updateOptoTrialTypeSequence = 1;
        BpodSystem.Data.PreviousOnFraction = S.GUI.OnFraction;
    end

    if updateOptoTrialTypeSequence
        numTrialsAddedToSequence = 0;
        OptoTrialTypesToAdd = [];
        switch S.GUI.OptoTrialTypeSeq
            case 1 % interleaved - random
                numOnTrialsToAdd = [];
                OptoTrialTypesToAdd = [repmat(1, 1, S.GUI.MaxImg - currentTrial + 1)];
                numOnTrialsToAdd = ceil(S.GUI.OnFraction * length(OptoTrialTypesToAdd));
                numOnTrialsToAdd_idxs = randperm(length(OptoTrialTypesToAdd), numOnTrialsToAdd);
                OptoTrialTypesToAdd(numOnTrialsToAdd_idxs) = 2;
                OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd;
            case 2 % interleaved - random first trial type
                FirstTrialType = ceil(rand(1,1)*2);
                switch FirstTrialType
                    case 1 % on first
                        SecondTrialType = 2;
                    case 2 % off first
                        SecondTrialType = 1;
                end
                while numTrialsAddedToSequence < S.GUI.MaxImg
                    OptoTrialTypesToAdd = [OptoTrialTypesToAdd repmat(FirstTrialType, 1, S.GUI.NumOptoTrialsPerBlock) repmat(SecondTrialType, 1, S.GUI.NumOptoTrialsPerBlock)];
                    numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumOptoTrialsPerBlock;
                end
                OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd(1:length(OptoTrialTypes) - currentTrial + 1);
            case 3 % interleaved - short first block
                while numTrialsAddedToSequence < S.GUI.MaxImg
                    OptoTrialTypesToAdd = [OptoTrialTypesToAdd repmat(1, 1, S.GUI.NumOptoTrialsPerBlock) repmat(2, 1, S.GUI.NumOptoTrialsPerBlock)];
                    numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumOptoTrialsPerBlock;
                end
                OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd(1:length(OptoTrialTypes) - currentTrial + 1);
            case 4 % interleaved - long first block
                while numTrialsAddedToSequence < S.GUI.MaxImg
                    OptoTrialTypesToAdd = [OptoTrialTypesToAdd repmat(2, 1, S.GUI.NumOptoTrialsPerBlock) repmat(1, 1, S.GUI.NumOptoTrialsPerBlock)];
                    numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumOptoTrialsPerBlock;
                end
                OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd(1:length(OptoTrialTypes) - currentTrial + 1);
        end
    end
end



    end
end
