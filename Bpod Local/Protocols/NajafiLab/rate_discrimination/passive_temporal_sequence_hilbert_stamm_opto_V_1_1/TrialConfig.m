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
        obj, S)
    switch S.GUI.EnableOpto
        case 0
            OptoTypes = zeros(1:S.GUI.MaxImg);
        case 1
            OptoTypes = ones(1:S.GUI.MaxImg);
    end
    %     case 1
    %         OptoTypes = zeros(1, S.GUI.MaxImg);
    %     case 2
    %         OptoTypes = ones(1, S.GUI.MaxImg);
    %     case 3
    %         OptoTypes = randi([0, 1], 1, S.GUI.MaxImg);
    %     case 4
    %         b1 = zeros(1, S.GUI.MaxImg/4);
    %         b2 = ones(1, S.GUI.MaxImg/4);
    %         if (rand() < 0.5)
    %             OptoTypes = repmat([b1 b2], 1, 5);
    %         else
    %             OptoTypes = repmat([b2 b1], 1, 5);
    %         end
    %         OptoTypes = OptoTypes(1:S.GUI.MaxImg);
    % end
    % OptoTypes = ExtendSeq(obj, OptoTypes);
end


% print info
function DispExpInfo( ...
        obj, ImgSeqLabel, NormalTypes, FixJitterTypes, OddballTypes, ...
        currentTrial, VisStim)
    disp('----------------------------------------------------')
    ExperimenterTrialInfo.ImgIdx = currentTrial;
    if (ImgSeqLabel(currentTrial) == -1)
        ExperimenterTrialInfo.Img = 'Oddball';
    elseif (ImgSeqLabel(currentTrial) < -1)
        ExperimenterTrialInfo.Img = 'OrienChange';
    elseif (ImgSeqLabel(currentTrial) == 2)
        ExperimenterTrialInfo.Img = '0 deg';
    elseif (ImgSeqLabel(currentTrial) == 3)
        ExperimenterTrialInfo.Img = '45 deg';
    elseif (ImgSeqLabel(currentTrial) == 4)
        ExperimenterTrialInfo.Img = '90 deg';
    elseif (ImgSeqLabel(currentTrial) == 5)
        ExperimenterTrialInfo.Img = '135 deg';
    end
    if (NormalTypes(currentTrial) == 0)
        ExperimenterTrialInfo.Normal = 'Short';
    elseif (NormalTypes(currentTrial) < -1)
        ExperimenterTrialInfo.Normal = 'Long';
    end
    if (FixJitterTypes(currentTrial) == 0)
        ExperimenterTrialInfo.FixJitter = 'Fix';
    elseif (FixJitterTypes(currentTrial) < -1)
        ExperimenterTrialInfo.FixJitter = 'Jitter';
    end
    if (OddballTypes(currentTrial) == 0)
        ExperimenterTrialInfo.Oddball = 'Short';
    elseif (OddballTypes(currentTrial) < -1)
        ExperimenterTrialInfo.Oddball = 'Long';
    end
    ExperimenterTrialInfo.ISI = VisStim.Gray.Dur;
    strExperimenterTrialInfo = formattedDisplayText(ExperimenterTrialInfo,'UseTrueFalseForLogical',true);
    disp(strExperimenterTrialInfo);
end


    end
end