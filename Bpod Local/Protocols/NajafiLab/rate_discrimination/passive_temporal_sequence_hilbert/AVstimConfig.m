classdef AVstimConfig
    methods


%% Video utils


% generate grey image
function [VideoGrayFixed] = GenGreyImg( ...
        obj, Xsize, Ysize)
    if Ysize > Xsize
        gratingSize = [Ysize, Ysize];
    else
        gratingSize = [Xsize, Xsize];
    end
    gray = 0.5 * ones(gratingSize);
    gray = gray(1:Ysize, 1:Xsize);
    gray = gray * 255;
    VideoGrayFixed = repmat(gray, 1, 1, 2);
end


% generate grating image
function [VideoGrating, VideoGrayFixed] = GenStimImg( ...
        obj, ImgParams, Xsize, Ysize)
    if Ysize > Xsize
        gratingSize = [Ysize, Ysize];
    else
        gratingSize = [Xsize, Xsize];
    end
    pixPerCycle = 1 / ImgParams.spatialFreq;
    freqPerPixel = 1 / pixPerCycle;
    [x, y] = meshgrid(1:gratingSize(1), 1:gratingSize(2));
    gray = 0.5 * ones(gratingSize);
    sinGrating = gray + ImgParams.contrast/2 .* sin(freqPerPixel * 2 * pi * (cos(ImgParams.orientation * pi / 180) * x + sin(ImgParams.orientation * pi / 180) * y) + ImgParams.phase);
    sinGrating(sinGrating > 1) = 1;
    sinGrating(sinGrating < 0) = 0;
    sinGrating = sinGrating(1:Ysize, 1:Xsize);
    sinGrating = sinGrating * 255;
    VideoGrating = repmat(sinGrating, 1, 1, 2);
end


% get frames from video duration
function [Frames] = GetFrames( ...
        obj, FPS, Dur)
    Frames = convergent(FPS * Dur);
    if (mod(Frames, 2) ~= 0)
        Frames = Frames + 1;
    end
end


% get video duration from video
function [VideoDur] = GetVideoDur( ...
        obj, FPS, Video)
    VideoDur = length(Video) * (1/FPS);
end


% generate unit video from frames and image
function [UnitVideo] = GetUnitVideo( ...
        obj, Frame_Sync, Frames)
    % UnitVideo = [repmat([Frame_Sync Frame_Sync], 1, Frames/2)];
    UnitVideo = [repmat([Frame_Sync], 1, Frames)];
end


%% Define basic video stimuli


% determine trial specific image
function [StimIdx] = GetStimIdx( ...
        obj, S, TrialType)
    StimIdx = mod(TrialType - 1, 6) + 1;
end


% generate trial image
function [VisStim] = GetVisStimImg( ...
        obj, S, BpodSystem, FPS, VisStim, TrialType)
    VisStim.Img.GrayFrame_SyncW    = BpodSystem.PluginObjects.V.Videos{1}.Data(1);
    VisStim.Img.GrayFrame_SyncBlk  = BpodSystem.PluginObjects.V.Videos{1}.Data(2);
    VisStim.Img.GrayBlank          = BpodSystem.PluginObjects.V.Videos{1}.Data(3);
    if (mod(TrialType-1,5)+1 > 1)
        idx = mod(TrialType-1,5)+1;
        SampleGratingIdx = VisStim.GratingIdx(idx-1);
        VisStim.SampleGratingIdx = SampleGratingIdx;
    else
        VisStim.SampleGratingIdx = VisStim.SampleGratingIdx;
    end
    VisStim.Img.GratingFrame_SyncW = BpodSystem.PluginObjects.V.Videos{VisStim.SampleGratingIdx}.Data(1);
    VisStim.Img.GratingBlank       = BpodSystem.PluginObjects.V.Videos{VisStim.SampleGratingIdx}.Data(3);
    VisStim.Grating.Frames = GetFrames(obj, FPS, S.GUI.GratingDur_s);
    VisStim.Grating.Video  = GetUnitVideo(obj, VisStim.Img.GratingFrame_SyncW, VisStim.Grating.Frames);
    VisStim.Grating.Dur    = GetVideoDur(obj, FPS, VisStim.Grating.Video);
end


% get a fix isi or sample a jitter isi
function [ISI] = GetFixJitterISI( ...
        obj, S, NormalType, FixJitterType)
    if (NormalType == 0)
        FixISI = S.GUI.ShortNormISI;
        JitterISI = 19961106;
        while (JitterISI < S.GUI.ShortRandomMin || ...
               JitterISI > S.GUI.ShortRandomMax)
            JitterISI = normrnd(S.GUI.ShortNormISI, S.GUI.ShortRandomStd);
        end
    elseif (NormalType == 1)
        FixISI = S.GUI.LongNormISI;
        JitterISI = 19961106;
        while (JitterISI < S.GUI.LongRandomMin || ...
               JitterISI > S.GUI.LongRandomMax)
            JitterISI = normrnd(S.GUI.LongNormISI, S.GUI.LongRandomStd);
        end
    end
    if (FixJitterType == 0)
        ISI = FixISI;
    else
        ISI = JitterISI;
    end
end


% get oddball ISI
function [ISI] = GetOddballISI( ...
        obj, S, NormalType, OddballType)
    if (NormalType == 0 && OddballType == 0)
        ISI = S.GUI.ShortNormShortOdd;
    elseif (NormalType == 0 && OddballType == 1)
        ISI = S.GUI.ShortNormLongOdd;
    elseif (NormalType == 1 && OddballType == 0)
        ISI = S.GUI.LongNormShortOdd;
    elseif (NormalType == 1 && OddballType == 1)
        ISI = S.GUI.LongNormLongOdd;
    end
end

% get isi sequence for all stim
function [ISIseq] = GetISIseq( ...
        obj, S, TrialTypes, NormalTypes, FixJitterTypes, OddballTypes)
    ISIseq = [];
    for i = 1:S.GUI.MaxImg
        if (TrialTypes(i) > 1)
            ISI = GetFixJitterISI(obj, S, NormalTypes(i), FixJitterTypes(i));
        else
            ISI = GetOddballISI(obj, S, NormalTypes(i), OddballTypes(i));
        end
        ISIseq = [ISIseq ISI];
    end
end

% get pre and post isi for the current stim
function [PreISI, OddISI, PostISI] = GetPrePostISIOdd( ...
        obj, ISIseq, currentImg)
    paddedISIseq = [ISIseq(1), ISIseq, ISIseq(end)];
    PreISI  = paddedISIseq(currentImg);
    OddISI  = paddedISIseq(currentImg + 1);
    PostISI = paddedISIseq(currentImg + 2);
    PreISI  = PreISI / 2;
    PostISI = PostISI / 2;
end

% get pre and post isi for the current stim
function [PreISI, PostISI] = GetPrePostISI( ...
        obj, ISIseq, currentImg)
    paddedISIseq = [ISIseq(1), ISIseq, ISIseq(end)];
    PreISI  = paddedISIseq(currentImg);
    PostISI = paddedISIseq(currentImg + 1);
    PreISI  = PreISI / 2;
    PostISI = PostISI / 2;
end

% generate single trial video for normal
function [VisStim] = GetVideoDataNormal( ...
        obj, S, BpodSystem, FPS, VisStim, TrialTypes, ISIseq, currentImg)
    VisStim = GetVisStimImg( ...
        obj, S, BpodSystem, FPS, VisStim, TrialTypes(currentImg));
    [PreISI, PostISI] = GetPrePostISI(obj, ISIseq, currentImg);
    % pre
    VisStim.PreGray.Frames = GetFrames(obj, FPS, PreISI);       
    VisStim.PreGray.Video  = [VisStim.Img.GrayFrame_SyncW, GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.PreGray.Frames-1)];
    VisStim.PreGray.Dur    = GetVideoDur(obj, FPS, VisStim.PreGray.Video);
    % oddball
    VisStim.OddballGray.Frames = nan;       
    VisStim.OddballGray.Video  = nan;
    VisStim.OddballGray.Dur    = nan;
    % post
    VisStim.PostGray.Frames = GetFrames(obj, FPS, PostISI);       
    VisStim.PostGray.Video  = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.PostGray.Frames);
    VisStim.PostGray.Dur    = GetVideoDur(obj, FPS, VisStim.PostGray.Video);
    % combine grey-grating-grey
    VisStim.Data.Video  = [VisStim.PreGray.Video, VisStim.Grating.Video, VisStim.PostGray.Video];
    VisStim.Data.Dur    = GetVideoDur(obj, FPS, VisStim.Data.Video);
    VisStim.Data.Frames = GetFrames(obj, FPS, VisStim.Data.Dur);
    VisStim.Data.Full = VisStim.Data.Video;
end

% generate single trial video for oddball
function [VisStim] = GetVideoDataOddball( ...
        obj, S, BpodSystem, FPS, VisStim, TrialTypes, ISIseq, currentImg)
    VisStim = GetVisStimImg( ...
        obj, S, BpodSystem, FPS, VisStim, TrialTypes(currentImg));
    [PreISI, OddISI, PostISI] = GetPrePostISIOdd(obj, ISIseq, currentImg);
    % pre
    VisStim.PreGray.Frames = GetFrames(obj, FPS, PreISI);       
    VisStim.PreGray.Video  = [VisStim.Img.GrayFrame_SyncW, GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.PreGray.Frames-1)];
    VisStim.PreGray.Dur    = GetVideoDur(obj, FPS, VisStim.PreGray.Video);
    % oddball
    VisStim.OddballGray.Frames = GetFrames(obj, FPS, OddISI);       
    VisStim.OddballGray.Video  = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.OddballGray.Frames);
    VisStim.OddballGray.Dur    = GetVideoDur(obj, FPS, VisStim.OddballGray.Video);
    % post
    VisStim.PostGray.Frames = GetFrames(obj, FPS, PostISI);       
    VisStim.PostGray.Video  = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.PostGray.Frames);
    VisStim.PostGray.Dur    = GetVideoDur(obj, FPS, VisStim.PostGray.Video);
    % combine grey-grating-grey-grating-grey
    VisStim.Data.Video  = [VisStim.PreGray.Video, VisStim.Grating.Video, VisStim.OddballGray.Video, VisStim.Grating.Video, VisStim.PostGray.Video];
    VisStim.Data.Dur    = GetVideoDur(obj, FPS, VisStim.Data.Video);
    VisStim.Data.Frames = GetFrames(obj, FPS, VisStim.Data.Dur);
    VisStim.Data.Full = VisStim.Data.Video;
end

% generate video sequence for all trials
function VisStimSeq = GetVisStimSeq( ...
        obj, S, BpodSystem, FPS, VisStim, TrialTypes, ISIseq)
    VisStimSeq = [];
    currentImg = 1;
    while currentImg <= S.GUI.MaxImg
        if TrialTypes(currentImg) ~= 1
            [VisStim] = GetVideoDataNormal( ...
                obj, S, BpodSystem, FPS, VisStim, TrialTypes, ISIseq, currentImg);
            VisStim.OddballFlag = 0;
            currentImg = currentImg + 1;
        elseif TrialTypes(currentImg) == 1
            [VisStim] = GetVideoDataOddball( ...
                obj, S, BpodSystem, FPS, VisStim, TrialTypes, ISIseq, currentImg);
            VisStim.OddballFlag = 1;
            currentImg = currentImg + 2;
        end
        VisStimSeq = [VisStimSeq VisStim];
    end
end


%% Audio


function [SoundWithEnvelope] = ApplySoundEnvelope( ...
        obj, Sound, Envelope)
    BackOfTheEnvelope = fliplr(Envelope);
    IdxsBetweenTheEnvelope = length(Sound) - 2 * length(Envelope);
    FullEnvelope = [Envelope ones(1, IdxsBetweenTheEnvelope) BackOfTheEnvelope];
    SoundWithEnvelope = Sound .* FullEnvelope;
end

function [Sound] = GenSinSound( ...
        obj, Freq, Dur, Vol, SF, Envelope)
    if (Dur == 0)
        Sound = [0];
    else
        Sound = GenerateSineWave(SF, Freq, Dur)*Vol;
        Sound = ApplySoundEnvelope(obj, Sound, Envelope);
    end
end

% generate audio sequence for all trials
function AudStimSeq = GetAudStimSeq( ...
        obj, S, SF, Envelope, TrialTypes, ISIseq)
    AudStimSeq = [];
    currentImg = 1;
    Grating = GenSinSound( ...
        obj, S.GUI.AudioStimFreq_Hz, S.GUI.GratingDur_s, S.GUI.AudioStimVolume_percent, SF, Envelope);
    while currentImg <= S.GUI.MaxImg
        if TrialTypes(currentImg) ~= 1
            [PreISI, PostISI] = GetPrePostISI(obj, ISIseq, currentImg);
            PreGrey  = zeros(1, ceil(PreISI * SF));
            PostGrey = zeros(1, ceil(PostISI * SF));
            AudStim.Data = [PreGrey Grating PostGrey];
            currentImg = currentImg + 1;
        elseif TrialTypes(currentImg) == 1
            [PreISI, OddISI, PostISI] = GetPrePostISIOdd(obj, ISIseq, currentImg);
            PreGrey  = zeros(1, ceil(PreISI * SF));
            OddGrey  = zeros(1, ceil(OddISI * SF));
            PostGrey = zeros(1, ceil(PostISI * SF));
            AudStim.Data = [PreGrey Grating OddGrey Grating PostGrey];
            currentImg = currentImg + 2;
        end
        AudStimSeq = [AudStimSeq AudStim];
    end
end


%% enable vis/aud stim


function [StimAct] = GetStimAct( ...
        obj, S)
    if S.GUI.VisStimEnable && S.GUI.AudioStimEnable
        StimAct = {'BNC2', 0};
    elseif S.GUI.VisStimEnable
        StimAct = {'HiFi1', ['P' 1]};
    elseif S.GUI.AudioStimEnable
        StimAct = {};
    else
        StimAct = {'HiFi1', ['P' 1]};
    end
end


function [AudStim] = GetAudStim( ...
        obj, S, OptoTrialType)
    switch S.GUI.EnableOpto
        case 0
            AudStim = {'HiFi1', ['P', 4]};
        case 1
            if OptoTrialType == 2
                AudStim = {'HiFi1', ['P', 4], 'GlobalTimerTrig', 1};
            else
                AudStim = {'HiFi1', ['P', 4]};
            end
    end
end


    end
end        