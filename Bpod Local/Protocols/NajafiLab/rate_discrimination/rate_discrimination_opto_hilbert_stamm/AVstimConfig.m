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
    UnitVideo = [repmat([Frame_Sync Frame_Sync], 1, Frames/2)];
end


% image random omission trigger flag
function [OmiFlag, LastOmiGrating] = GetOmiFlag( ...
        obj, S, CurrentRep, LastOmiGrating)
    OmiFlag = 'False';
    r = rand();
    if (S.GUI.ActOmi == 1 && ...
        abs(CurrentRep-LastOmiGrating) > S.GUI.OmiMinInterval && ...
        CurrentRep > S.GUI.OmiAvoidFrame && ...
        CurrentRep < S.GUI.PrePertFlashRep-S.GUI.OmiAvoidFrame && ...
        r < S.GUI.OmiProb)
            OmiFlag = 'True';
            LastOmiGrating = CurrentRep;
    end
end


% sampling a random isi
function ISI = GetRandomISI( ...
        obj, S, ISICenter)
    ISI = 19961106;
    while (ISI < S.GUI.RandomISIMin || ...
           ISI > S.GUI.RandomISIMax || ...
           ISI < ISICenter-S.GUI.RandomISIWid || ...
           ISI > ISICenter+S.GUI.RandomISIWid)
        ISI = normrnd(ISICenter, S.GUI.RandomISIStd);
    end
end


%% Define basic video stimuli


function [VisStim] = GetVisStimImg( ...
        obj, S, BpodSystem, FPS, VisStim)
    VisStim.Img.GrayFrame_SyncW    = BpodSystem.PluginObjects.V.Videos{1}.Data(1);
    VisStim.Img.GrayFrame_SyncBlk  = BpodSystem.PluginObjects.V.Videos{1}.Data(2);
    VisStim.Img.GrayBlank          = BpodSystem.PluginObjects.V.Videos{1}.Data(3);
    if (S.GUI.RandomOrient == 1)
        idx = VisStim.GratingIdx( randperm(length(VisStim.GratingIdx),1) );
    else
        idx = VisStim.GratingIdx(1);
    end
    VisStim.Img.GratingFrame_SyncW = BpodSystem.PluginObjects.V.Videos{idx}.Data(1);
    VisStim.Img.GratingBlank       = BpodSystem.PluginObjects.V.Videos{idx}.Data(3);
    VisStim.Grating.Frames = GetFrames(obj, FPS, S.GUI.GratingDur_s);
    VisStim.Grating.Video  = GetUnitVideo(obj, VisStim.Img.GratingFrame_SyncW, VisStim.Grating.Frames);
    VisStim.Grating.Dur    = GetVideoDur(obj, FPS, VisStim.Grating.Video);
end


function [VisStim] = GetVideoDataPre( ...
        obj, S, BpodSystem, FPS, VisStim, RandomISI)
    VideoSeq = [];
    PreISIinfo = [];
    LastOmiGrating = 0;
    for CurrentRep = 1:S.GUI.PrePertFlashRep
        [VisStim] = GetVisStimImg( ...
            obj, S, BpodSystem, FPS, VisStim);
        if (RandomISI == 1 && S.GUI.ActRandomISI == 1)
            ISI = GetRandomISI(obj, S, S.GUI.ISIOrig_s);
        else
            ISI = S.GUI.ISIOrig_s;
        end
        [OmiFlag, LastOmiGrating] = GetOmiFlag( ...
            obj, S, CurrentRep, LastOmiGrating);
        switch OmiFlag
            case 'True'
                ISI = 2 * S.GUI.ISIOrig_s;
                VisStim.GrayPre.Frames      = GetFrames(obj, FPS, ISI);
                VisStim.GrayPre.Video       = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.GrayPre.Frames);
                VisStim.GrayPre.Dur         = GetVideoDur(obj, FPS, VisStim.GrayPre.Video);
                VisStim.Data.PreUnit.Video  = [VisStim.Grating.Video, VisStim.GrayPre.Video];
                Seq = [ones(1, length(VisStim.Grating.Video)), zeros(1, length(VisStim.GrayPre.Video))];
                Seq(length(VisStim.Grating.Video)+ceil(length(VisStim.GrayPre.Video)/2)) = -1;
            case 'False'
                VisStim.GrayPre.Frames      = GetFrames(obj, FPS, ISI);
                VisStim.GrayPre.Video       = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.GrayPre.Frames);
                VisStim.GrayPre.Dur         = GetVideoDur(obj, FPS, VisStim.GrayPre.Video);
                VisStim.Data.PreUnit.Video  = [VisStim.Grating.Video, VisStim.GrayPre.Video];
                Seq = [ones(1, length(VisStim.Grating.Video)), zeros(1, length(VisStim.GrayPre.Video))];
        end
        VisStim.Data.PreUnit.Dur    = GetVideoDur(obj, FPS, VisStim.Data.PreUnit.Video);
        VisStim.Data.PreUnit.Frames = GetFrames(obj, FPS, VisStim.Data.PreUnit.Dur);
        VideoSeq = [VideoSeq, VisStim.Data.PreUnit.Video];
        PreISIinfo = [PreISIinfo, VisStim.GrayPre.Dur];
        VisStim.ProcessedData.Seq = [VisStim.ProcessedData.Seq, Seq];
        VisStim.ProcessedData.PrePost = [VisStim.ProcessedData.PrePost, zeros(1, length(VisStim.Data.PreUnit.Video))];
    end
    VideoSeq = [VideoSeq, VisStim.Grating.Video];
    VisStim.ProcessedData.Seq = [VisStim.ProcessedData.Seq, ones(1, length(VisStim.Grating.Video))];
    VisStim.ProcessedData.PrePost = [VisStim.ProcessedData.PrePost, zeros(1, length(VisStim.Grating.Video))];
    VisStim.Data.Pre.Video  = VideoSeq;
    VisStim.Data.Pre.Dur    = GetVideoDur(obj, FPS, VisStim.Data.Pre.Video);
    VisStim.Data.Pre.Frames = GetFrames(obj, FPS, VisStim.Data.Pre.Dur);
    VisStim.PreISIinfo = PreISIinfo;
    VisStim.OmiFlag    = OmiFlag;
end


function [VisStim] = GetVideoDataPost( ...
        obj, S, BpodSystem, FPS, VisStim, GrayPerturbISI, RandomISI)
    VideoSeq = [];
    PostISIinfo = [];
    while (GetVideoDur(obj, FPS, VideoSeq) <= (S.GUI.PostPertDur))
        [VisStim] = GetVisStimImg( ...
            obj, S, BpodSystem, FPS, VisStim);
        if (RandomISI == 1 && S.GUI.ActRandomISI == 1)
            ISI = GetRandomISI(obj, S, GrayPerturbISI);
        else
            ISI = GrayPerturbISI;
        end       
        VisStim.GrayPost.Frames = GetFrames(obj, FPS, ISI);
        VisStim.GrayPost.Video  = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.GrayPost.Frames);
        VisStim.GrayPost.Dur    = GetVideoDur(obj, FPS, VisStim.GrayPost.Video);
        VisStim.Data.PostUnit.Video  = [VisStim.GrayPost.Video VisStim.Grating.Video];
        VisStim.Data.PostUnit.Dur    = GetVideoDur(obj, FPS, VisStim.Data.PostUnit.Video);
        VisStim.Data.PostUnit.Frames = GetFrames(obj, FPS, VisStim.Data.PostUnit.Dur);
        Seq = [zeros(1, length(VisStim.GrayPost.Video)) ones(1, length(VisStim.Grating.Video))];
        VisStim.ProcessedData.Seq = [VisStim.ProcessedData.Seq, Seq];
        VisStim.ProcessedData.PrePost = [VisStim.ProcessedData.PrePost, ones(1, length(VisStim.Data.PostUnit.Video))];
        VideoSeq = [VideoSeq, VisStim.Data.PostUnit.Video];
        PostISIinfo = [PostISIinfo, VisStim.GrayPost.Dur];
    end
    VisStim.Data.Post.Video  = VideoSeq;
    VisStim.Data.Post.Dur    = GetVideoDur(obj, FPS, VisStim.Data.Post.Video);
    VisStim.Data.Post.Frames = GetFrames(obj, FPS, VisStim.Data.Post.Dur);
    VisStim.PostISIinfo = PostISIinfo;
end


function [VisStim] = GetVideoDataExtra( ...
        obj, S, BpodSystem, FPS, VisStim, GrayPerturbISI, RandomISI)
    VisStim.Data.Extra.Dur = GetPostPertDurExtra(obj, S);
    if (VisStim.Data.Extra.Dur > 0)
        VideoSeq = [];
        ExtraISIinfo = [];
        while (GetVideoDur(obj, FPS, VideoSeq) <= VisStim.Data.Extra.Dur)
            [VisStim] = GetVisStimImg( ...
                obj, S, BpodSystem, FPS, VisStim);
            if (RandomISI == 1 && S.GUI.ActRandomISI == 1)
                ISI = GetRandomISI(obj, S, GrayPerturbISI);
            else
                ISI = GrayPerturbISI;
            end
            VisStim.GrayExtra.Frames = GetFrames(obj, FPS, ISI);
            VisStim.GrayExtra.Video  = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.GrayExtra.Frames);
            VisStim.GrayExtra.Dur    = GetVideoDur(obj, FPS, VisStim.GrayExtra.Video);
            VisStim.Data.ExtraUnit.Video  = [VisStim.GrayExtra.Video VisStim.Grating.Video];
            VisStim.Data.ExtraUnit.Dur    = GetVideoDur(obj, FPS, VisStim.Data.ExtraUnit.Video);
            VisStim.Data.ExtraUnit.Frames = GetFrames(obj, FPS, VisStim.Data.ExtraUnit.Dur);
            Seq = [zeros(1, length(VisStim.GrayExtra.Video)) ones(1, length(VisStim.Grating.Video))];
            VisStim.ProcessedData.Seq = [VisStim.ProcessedData.Seq, Seq];
            VisStim.ProcessedData.PrePost = [VisStim.ProcessedData.PrePost, 2*ones(1, length(VisStim.Data.ExtraUnit.Video))];
            VideoSeq = [VideoSeq, VisStim.Data.ExtraUnit.Video];
            ExtraISIinfo = [ExtraISIinfo, VisStim.GrayExtra.Dur];
        end
        VisStim.Data.Extra.Video  = VideoSeq;
        VisStim.Data.Extra.Dur    = GetVideoDur(obj, FPS, VisStim.Data.Extra.Video);
        VisStim.Data.Extra.Frames = GetFrames(obj, FPS, VisStim.Data.Extra.Dur);
        VisStim.ExtraISIinfo = ExtraISIinfo;
    else
        VisStim.Data.Extra.Video = [];
        VisStim.Data.Extra.Dur = 0;
        VisStim.Data.Extra.Frames = 0;
        VisStim.ExtraISIinfo = [];
    end
end


function [VideoData] = GetFullVideo( ...
        obj, VisStim)
    VideoData = [VisStim.Data.Pre.Video VisStim.Data.Post.Video VisStim.Data.Extra.Video];
end


%% Audio utils


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


function [Sound] = GenWhiteNoise( ...
        obj, Dur, Vol, SF, Envelope)
    if (Dur == 0)
        Sound = [0];
    else
        Sound = GenerateWhiteNoise(SF, Dur, 1, 1)*Vol; 
        Sound = ApplySoundEnvelope(obj, Sound, Envelope);
    end
end


%% Define audio stimuli


function ConfigHifi( ...
        obj, H, S, SF, Envelope)
    [InitCueSound] = GenSinSound( ...
        obj, S.GUI.InitCueFreq_Hz, S.GUI.InitCueDuration_s, S.GUI.InitCueVolume_percent, SF, Envelope);
    [GoCueSound] = GenSinSound( ...
        obj, S.GUI.GoCueFreq_Hz, S.GUI.GoCueDuration_s, S.GUI.GoCueVolume_percent, SF, Envelope);
    [IncorrectSound] = GenWhiteNoise( ...
        obj, S.GUI.NoiseDuration_s, S.GUI.NoiseVolume_percent, SF, Envelope);
    H.load(1, InitCueSound);
    H.load(2, GoCueSound);
    H.load(3, IncorrectSound);
end


%% build audio stim to match grating/gray pattern


function [FullAudio] = GenAudioStim( ...
        obj, S, VisStim, SF, Envelope)
    AudioStimSound = GenSinSound( ...
        obj, S.GUI.AudioStimFreq_Hz, S.GUI.GratingDur_s, S.GUI.AudioStimVolume_percent, SF, Envelope);
    [GoCueSound] = GenSinSound( ...
        obj, S.GUI.GoCueFreq_Hz, S.GUI.GoCueDuration_s, S.GUI.GoCueVolume_percent, SF, Envelope);
    AudioSeq = [];
    if ~isempty(VisStim.PreISIinfo)
        for i = 1 : length(VisStim.PreISIinfo)
            GrayNumSamples = ceil(VisStim.PreISIinfo(i) * SF);
            NoSoundPrePert = zeros(1, GrayNumSamples);
            AudioSeq = [AudioSeq AudioStimSound NoSoundPrePert];
        end
        AudioSeq = [AudioSeq AudioStimSound];
    end
    if ~isempty(VisStim.PostISIinfo)
        for i = 1 : length(VisStim.PostISIinfo)
            GrayNumSamples = ceil(VisStim.PostISIinfo(i) * SF);
            NoSoundPrePert = zeros(1, GrayNumSamples);
            AudioSeq = [AudioSeq NoSoundPrePert AudioStimSound];
        end
    end
    if ~isempty(VisStim.ExtraISIinfo)
        StartIdx = length(AudioSeq);
        for i = 1 : length(VisStim.ExtraISIinfo)-1
            GrayNumSamples = ceil(VisStim.ExtraISIinfo(i) * SF);
            NoSoundPrePert = zeros(1, GrayNumSamples);
            AudioSeq = [AudioSeq NoSoundPrePert AudioStimSound];
        end
    end
    if (S.GUI.EnablePassive==0)
        GoCueStartIdx = ceil((VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur) * SF);
        AudioSeq = [AudioSeq, zeros(1, length(GoCueSound))];
        AudioSeq(GoCueStartIdx+1:GoCueStartIdx+length(GoCueSound)) = AudioSeq(GoCueStartIdx+1:GoCueStartIdx+length(GoCueSound)) + GoCueSound;
    end
    FullAudio = AudioSeq;
end


%% Define trial-specific video stim


function [PostPertDurExtra] = GetPostPertDurExtra( ...
        obj, S)
    switch S.GUI.ExtraStim
        case 1 % default
            switch S.GUI.TrainingLevel
                case 1 % passive
                    PostPertDurExtra = 0;
                case 2 % Naive
                    PostPertDurExtra = 30;
                case 3 % Mid Trained 1
                    PostPertDurExtra = 20;
                case 4 % Mid Trained 2
                    PostPertDurExtra = 1;
                case 5 % Trained
                    PostPertDurExtra = 0;
            end
        case 2 % manually extra
            PostPertDurExtra = S.GUI.PostPertDurExtra;
        case 3 % manually no extra
            PostPertDurExtra = 0;
    end
end


function ConfigFullAudioStim( ...
        obj, H, S, VisStim, SF, Envelope)
    [FullAudio] = GenAudioStim( ...
        obj, S, VisStim, SF, Envelope);
    H.load(5, FullAudio);
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






    end
end        