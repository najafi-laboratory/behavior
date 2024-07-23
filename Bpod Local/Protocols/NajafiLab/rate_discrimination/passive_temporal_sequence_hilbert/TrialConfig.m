classdef TrialConfig
    methods


% extend type sequence to prevent crash
function [ExtendedSeq] = ExtendSeq( ...
        obj, Seq)
    ExtendedSeq = repmat(Seq, 1, 43);
end


% generate a sequence with orientation mini blocks and oddballs
function [TrialTypes, ImgSeqLabel] = GenTrialTypesSeq( ...
        obj, S)
    TrialTypes = [];
    ImgSeqLabel = [];
    possibleTargetImg = [2, 3, 4, 5];
    while length(TrialTypes) < S.GUI.MaxImg
        TargetImg = possibleTargetImg(randi(length(possibleTargetImg)));
        if (~isempty(TrialTypes))
            while (TargetImg == TrialTypes(end))
                TargetImg = possibleTargetImg(randi(length(possibleTargetImg)));
            end
        end
        SequenceBlock = repmat(TargetImg, 1, randi([S.GUI.OrienBlockNumMin, S.GUI.OrienBlockNumMax]));
        indicesToChange = rand(size(SequenceBlock)) < S.GUI.OddProb;
        indicesToChange(1:S.GUI.OddAvoidFrameStart) = false;
        indicesToChange(end-S.GUI.OddAvoidFrameEnd+1:end) = false;
        for i = S.GUI.OddAvoidFrameStart+1:length(indicesToChange)-S.GUI.OddAvoidFrameEnd
            if indicesToChange(i)
                indicesToChange(i+1:i+S.GUI.OddAvoidFrameBetween) = false;
            end
        end
        SequenceBlock(indicesToChange) = 1;
        TrialTypes = [TrialTypes, SequenceBlock];
        if (~isempty(ImgSeqLabel) && ImgSeqLabel(end) ~= SequenceBlock(1))
            SequenceBlock(1) = -SequenceBlock(1);
        end
        ImgSeqLabel = [ImgSeqLabel SequenceBlock];
    end
    TrialTypes = TrialTypes(1:S.GUI.MaxImg);
    ImgSeqLabel(ImgSeqLabel==1) = -1;
    ImgSeqLabel = ImgSeqLabel(1:S.GUI.MaxImg);
    TrialTypes = ExtendSeq(obj, TrialTypes);
    ImgSeqLabel = ExtendSeq(obj, ImgSeqLabel);
end


% generaate a sequence with short long baseline block
function [NormalTypes] = GenNormalTypes( ...
        obj, S)
    switch S.GUI.NormalMode
        case 1
            NormalTypes = zeros(1, S.GUI.MaxImg);
        case 2
            NormalTypes = ones(1, S.GUI.MaxImg);
        case 3
            NormalTypes = randi([0, 1], 1, S.GUI.MaxImg);
        case 4
            b1 = zeros(1, S.GUI.MaxImg/4);
            b2 = ones(1, S.GUI.MaxImg/4);
            if (rand() < 0.5)
                NormalTypes = repmat([b1 b2], 1, 5);
            else
                NormalTypes = repmat([b2 b1], 1, 5);
            end
            NormalTypes = NormalTypes(1:S.GUI.MaxImg);
    end
    NormalTypes = ExtendSeq(obj, NormalTypes);
end


% generaate a sequence with fix jitter block
function [FixJitterTypes] = GenFixJitterTypes( ...
        obj, S)
    switch S.GUI.FixJitterMode
        case 1
            FixJitterTypes = zeros(1, S.GUI.MaxImg);
        case 2
            FixJitterTypes = ones(1, S.GUI.MaxImg);
        case 3
            FixJitterTypes = randi([0, 1], 1, S.GUI.MaxImg);
        case 4
            b1 = zeros(1, S.GUI.MaxImg/4);
            b2 = ones(1, S.GUI.MaxImg/4);
            if (rand() < 0.5)
                FixJitterTypes = repmat([b1 b2], 1, 5);
            else
                FixJitterTypes = repmat([b2 b1], 1, 5);
            end
            FixJitterTypes = FixJitterTypes(1:S.GUI.MaxImg);
    end
    FixJitterTypes = ExtendSeq(obj, FixJitterTypes);
end


% generaate a sequence with fix jitter block
function [OddballTypes] = GenOddballTypes( ...
        obj, S)
    switch S.GUI.OddballMode
        case 1
            OddballTypes = zeros(1, S.GUI.MaxImg);
        case 2
            OddballTypes = ones(1, S.GUI.MaxImg);
        case 3
            OddballTypes = randi([0, 1], 1, S.GUI.MaxImg);
        case 4
            b1 = zeros(1, S.GUI.MaxImg/4);
            b2 = ones(1, S.GUI.MaxImg/4);
            if (rand() < 0.5)
                OddballTypes = repmat([b1 b2], 1, 5);
            else
                OddballTypes = repmat([b2 b1], 1, 5);
            end
            OddballTypes = OddballTypes(1:S.GUI.MaxImg);
    end
    OddballTypes = ExtendSeq(obj, OddballTypes);
end


% generaate a sequence with short long baseline block
function [OptoTypes] = GenOptoTypes( ...
        obj, S, TrialTypes)
    switch S.GUI.OptoMode
        case 1
            OptoTypes = zeros(1, S.GUI.MaxImg);
        case 2
            OptoTypes = ones(1, S.GUI.MaxImg);
        case 3
            OptoTypes = zeros(size(TrialTypes));
            % oddball
            IdxOdd = find(TrialTypes == 1);
            NumPicked = round(S.GUI.OptoProb * length(IdxOdd));
            IdxOddPicked = randsample(IdxOdd, NumPicked);
            OptoTypes(IdxOddPicked) = 1;
            % post oddball
            IdxPostOdd = IdxOdd + 1;
            ValidIdxPostOdd = IdxPostOdd(ismember(TrialTypes(IdxPostOdd), [2, 3, 4, 5]));
            NotIdxOddPicked = setdiff(ValidIdxPostOdd, IdxOddPicked + 1);
            NumPicked = round(S.GUI.OptoProb * length(NotIdxOddPicked));
            IdxPostOddPicked = randsample(NotIdxOddPicked, NumPicked);
            OptoTypes(IdxPostOddPicked) = 2;
            % normal
            IdxNormal = find(TrialTypes == 2 | TrialTypes == 3 | TrialTypes == 4 | TrialTypes == 5);
            notAfter1 = IdxNormal(~ismember(IdxNormal, IdxOdd + 1));
            NumPicked = round(S.GUI.OptoProb * length(notAfter1));
            IdxNormalPicked = randsample(notAfter1, NumPicked);
            OptoTypes(IdxNormalPicked) = 3;
            % interval
            nonZeroIndices = find(OptoTypes ~= 0);
            for i = 2:length(nonZeroIndices)
                if nonZeroIndices(i) - nonZeroIndices(i-1) <= S.GUI.OptoAvoidFrameBetween
                    OptoTypes(nonZeroIndices(i)) = 0;
                end
            end
        case 4
            OptoTypes = zeros(1, S.GUI.MaxImg);
            i = randsample(S.GUI.MaxImg, round(S.GUI.OptoProb * S.GUI.MaxImg));
            OptoTypes(i) = 1;
        case 5
            b1 = zeros(1, S.GUI.MaxImg/4);
            b2 = ones(1, S.GUI.MaxImg/4);
            if (rand() < 0.5)
                OptoTypes = repmat([b1 b2], 1, 5);
            else
                OptoTypes = repmat([b2 b1], 1, 5);
            end
            OptoTypes = OptoTypes(1:S.GUI.MaxImg);
    end
    OptoTypes = ExtendSeq(obj, OptoTypes);
    OptoTypes(1:S.GUI.OptoAvoidFrameStart) = 0;
end


    end
end