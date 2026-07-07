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
    Frames = round(FPS * Dur);
    % Frames = convergent(FPS * Dur);
    % if (mod(Frames, 2) ~= 0)
    %     Frames = Frames + 1;
    % end
end


% get video duration from video
function [VideoDur] = GetVideoDur( ...
        obj, FPS, Video)
    VideoDur = length(Video) * (1/FPS);
end


% generate unit video from frames and image
function [UnitVideo] = GetUnitVideo( ...
        obj, Frame_Sync, Frames)
    UnitVideo = [repmat([Frame_Sync], 1, Frames)];
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
    % use grating index to get vis stim of corresponding angular
    % orientation
    % VisStim.orientation = [0 45 90 135];
    % VisStim.GratingIdx = [2 3 4 5];
    if (S.GUI.RandomOrient == 1)
        SampleGratingIdx = VisStim.GratingIdx( randperm(length(VisStim.GratingIdx),1) );
    else
        SampleGratingIdx = VisStim.GratingIdx(1);
    end
    VisStim.SampleGratingIdx = SampleGratingIdx;
    VisStim.Img.GratingFrame_SyncW = BpodSystem.PluginObjects.V.Videos{SampleGratingIdx}.Data(1);
    VisStim.Img.GratingFrame_SyncBlk = BpodSystem.PluginObjects.V.Videos{SampleGratingIdx}.Data(2);
    VisStim.Img.GratingBlank       = BpodSystem.PluginObjects.V.Videos{SampleGratingIdx}.Data(3);
    VisStim.Grating.Frames = GetFrames(obj, FPS, S.GUI.GratingDur_s);
    VisStim.Grating.Video  = GetUnitVideo(obj, VisStim.Img.GratingFrame_SyncW, VisStim.Grating.Frames);
    VisStim.Grating.Dur    = GetVideoDur(obj, FPS, VisStim.Grating.Video);
end


function [VisStim] = GetVideoDataPre( ...
        obj, S, BpodSystem, FPS, VisStim, RandomISI)
    VideoSeq = [];
    PreISIinfo = [];
    for CurrentRep = 1:S.GUI.PrePertFlashRep
        if (RandomISI == 1 && S.GUI.ActRandomISI == 1)
            ISI = GetRandomISI(obj, S, S.GUI.ISIOrig_s);
        else
            ISI = S.GUI.ISIOrig_s;
        end
        VisStim.GrayPre.Frames      = GetFrames(obj, FPS, ISI);
        VisStim.GrayPre.Video       = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.GrayPre.Frames);
        VisStim.GrayPre.Dur         = GetVideoDur(obj, FPS, VisStim.GrayPre.Video);
        VisStim.Data.PreUnit.Video  = [VisStim.Grating.Video, VisStim.GrayPre.Video];
        Seq = [ones(1, length(VisStim.Grating.Video)) * VisStim.SampleGratingIdx zeros(1, length(VisStim.GrayPre.Video))];
        VisStim.Data.PreUnit.Dur    = GetVideoDur(obj, FPS, VisStim.Data.PreUnit.Video);
        VisStim.Data.PreUnit.Frames = GetFrames(obj, FPS, VisStim.Data.PreUnit.Dur);
        VideoSeq = [VideoSeq, VisStim.Data.PreUnit.Video];
        PreISIinfo = [PreISIinfo, VisStim.GrayPre.Dur];
        VisStim.ProcessedData.Seq = [VisStim.ProcessedData.Seq, Seq];
        VisStim.ProcessedData.PrePost = [VisStim.ProcessedData.PrePost, zeros(1, length(VisStim.Data.PreUnit.Video))];
    end
    if S.GUI.PrePertFlashRep > 0

        VideoSeq = [VideoSeq, VisStim.Grating.Video];
        VisStim.ProcessedData.Seq = [VisStim.ProcessedData.Seq, ones(1, length(VisStim.Grating.Video)) * VisStim.SampleGratingIdx];
        VisStim.ProcessedData.PrePost = [VisStim.ProcessedData.PrePost, zeros(1, length(VisStim.Grating.Video))];
    else
        VideoSeq = [];
        VisStim.ProcessedData.Seq = [];
        VisStim.ProcessedData.PrePost = [];
    end
    VisStim.Data.Pre.Video  = VideoSeq;
    VisStim.Data.Pre.Dur    = GetVideoDur(obj, FPS, VisStim.Data.Pre.Video);
    VisStim.Data.Pre.Frames = GetFrames(obj, FPS, VisStim.Data.Pre.Dur);
    VisStim.PreISIinfo = PreISIinfo;
end


function [VisStim] = GetVideoDataPost( ...
        obj, S, BpodSystem, FPS, VisStim, GrayPerturbISI, RandomISI)
    VideoSeq = [];
    PostISIinfo = [];
    % might use loop later for fixed stim dur or number of images
    % while (GetVideoDur(obj, FPS, VideoSeq) <= (S.GUI.PostPertDur))
    
    % if pre pert, then consider image at perturbation boundary to be post
    % pert flash
    if S.GUI.PrePertFlashRep > 0
        PostPertFlashes = 1;
    else
        PostPertFlashes = 2;
    end
    for numImages = 1:PostPertFlashes
        if (RandomISI == 1 && S.GUI.ActRandomISI == 1)
            ISI = GetRandomISI(obj, S, GrayPerturbISI);
        else
            ISI = GrayPerturbISI;
        end
        VisStim.GrayPost.Frames = GetFrames(obj, FPS, ISI);
        VisStim.GrayPost.Video  = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, VisStim.GrayPost.Frames);
        VisStim.GrayPost.Dur    = GetVideoDur(obj, FPS, VisStim.GrayPost.Video);
        % VisStim.Data.PostUnit.Video  = [VisStim.Grating.Video VisStim.GrayPost.Video];
        if S.GUI.PrePertFlashRep > 0
            VisStim.Data.PostUnit.Video  = [VisStim.GrayPost.Video VisStim.Grating.Video VisStim.GrayPost.Video];
            Seq = [zeros(1, length(VisStim.GrayPost.Video)) ones(1, length(VisStim.Grating.Video)) * VisStim.SampleGratingIdx zeros(1, length(VisStim.GrayPost.Video))];
        else
            VisStim.Data.PostUnit.Video  = [VisStim.Grating.Video VisStim.GrayPost.Video];
            Seq = [ones(1, length(VisStim.Grating.Video)) * VisStim.SampleGratingIdx zeros(1, length(VisStim.GrayPost.Video))];
        end
        VisStim.Data.PostUnit.Dur    = GetVideoDur(obj, FPS, VisStim.Data.PostUnit.Video);
        VisStim.Data.PostUnit.Frames = GetFrames(obj, FPS, VisStim.Data.PostUnit.Dur);
        % Seq = [ones(1, length(VisStim.Grating.Video)) * VisStim.SampleGratingIdx zeros(1, length(VisStim.GrayPost.Video))];        
        VisStim.ProcessedData.Seq = [VisStim.ProcessedData.Seq, Seq];
        VisStim.ProcessedData.PrePost = [VisStim.ProcessedData.PrePost, ones(1, length(VisStim.Data.PostUnit.Video))];
        VideoSeq = [VideoSeq, VisStim.Data.PostUnit.Video];
        PostISIinfo = [PostISIinfo, VisStim.GrayPost.Dur];
    end
    VisStim.Data.Post.Video  = VideoSeq;
    VisStim.Data.Post.Dur    = GetVideoDur(obj, FPS, VisStim.Data.Post.Video);
    VisStim.Data.Post.Frames = GetFrames(obj, FPS, VisStim.Data.Post.Dur);
    VisStim.PostISIinfo = PostISIinfo;
    VisStim.ProcessedData.PostMeanISI = GrayPerturbISI;
end


function [VisStim] = GetVideoDataExtra( ...
        obj, S, BpodSystem, FPS, VisStim, GrayPerturbISI, RandomISI)
    VisStim.Data.Extra.Dur = GetPostPertDurExtra(obj, S);
    if (VisStim.Data.Extra.Dur > 0)
        VideoSeq = [];
        ExtraISIinfo = [];
        while (GetVideoDur(obj, FPS, VideoSeq) <= VisStim.Data.Extra.Dur)
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
            Seq = [zeros(1, length(VisStim.GrayExtra.Video)) ones(1, length(VisStim.Grating.Video)) * VisStim.SampleGratingIdx];
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
        obj, S, VisStim, FPS)
    % Frames = GetFrames(obj, FPS, S.GUI.ISIOrig_s);
    % EndGrey = GetUnitVideo(obj, VisStim.Img.GrayFrame_SyncBlk, Frames);
    % VideoData = [VisStim.Data.Pre.Video VisStim.Data.Post.Video VisStim.Data.Extra.Video EndGrey];
    VideoData = [VisStim.Data.Pre.Video VisStim.Data.Post.Video];
    % VideoData = VisStim.Data.Post.Video;
end


%% Audio utils


function [SoundWithEnvelope] = ApplySoundEnvelope( ...
        obj, Sound, Envelope)
    BackOfTheEnvelope = fliplr(Envelope);
    IdxsBetweenTheEnvelope = length(Sound) - 2 * length(Envelope);
    FullEnvelope = [Envelope ones(1, IdxsBetweenTheEnvelope) BackOfTheEnvelope];
    SoundWithEnvelope = Sound .* FullEnvelope;
    % SoundWithEnvelope = Sound;  % toggle envelope to align aud/vis
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
    AudioSeqPre = [];
    AudioSeqPost = [];
    AudioSeqExtra = [];
    if ~isempty(VisStim.PreISIinfo)
        for i = 1 : length(VisStim.PreISIinfo)
            GrayNumSamples = ceil(VisStim.PreISIinfo(i) * SF);
            NoSoundPrePert = zeros(1, GrayNumSamples);
            AudioSeqPre = [AudioSeqPre AudioStimSound NoSoundPrePert];
        end
        AudioSeqPre = [AudioSeqPre AudioStimSound];
    end
    if ~isempty(VisStim.PostISIinfo)
        for i = 1 : length(VisStim.PostISIinfo)
            GrayNumSamples = ceil(VisStim.PostISIinfo(i) * SF);
            NoSoundPostPert = zeros(1, GrayNumSamples);
            % if including pre pert, then post is ISI->grating pattern
            % otherwise its grating->ISI
            if S.GUI.PrePertFlashRep > 0
                AudioSeqPost = [AudioSeqPost NoSoundPostPert AudioStimSound];
            else
                AudioSeqPost = [AudioSeqPost AudioStimSound NoSoundPostPert];
            end
            
        end
    end
    % if ~isempty(VisStim.ExtraISIinfo)
    %     for i = 1 : length(VisStim.ExtraISIinfo)
    %         GrayNumSamples = ceil(VisStim.ExtraISIinfo(i) * SF);
    %         NoSoundPrePert = zeros(1, GrayNumSamples);
    %         AudioSeqExtra = [AudioSeqExtra NoSoundPrePert AudioStimSound];
    %     end
    % end

    % GrayNumSamples = ceil(S.GUI.ISIOrig_s * SF);
    % EndNoSound = zeros(1, GrayNumSamples);
    % AudioSeq = [AudioSeqPre AudioSeqPost AudioSeqExtra EndNoSound];
    % AudioSeq = AudioSeqPost;
    AudioSeq = [AudioSeqPre AudioSeqPost];
    
    % go cue
    % GoCueStartIdx = ceil((VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur) * SF);
    % AudioSeq = [AudioSeq, zeros(1, length(GoCueSound))];
    % AudioSeq(GoCueStartIdx+1:GoCueStartIdx+length(GoCueSound)) = AudioSeq(GoCueStartIdx+1:GoCueStartIdx+length(GoCueSound)) + GoCueSound;
    
    % opto delay
    % VideoOptoDelayDur = 0.032292;
    % OptoAudioStimOffsetNumSamples = VideoOptoDelayDur * SF;
    % OptoAudioStimOffset = zeros(1, floor(OptoAudioStimOffsetNumSamples));
    % FullAudio = [OptoAudioStimOffset AudioSeq];

    FullAudio = AudioSeq;
end


%% Define trial-specific video stim


function [PostPertDurExtra] = GetPostPertDurExtra( ...
        obj, S)
    switch S.GUI.ExtraStim
        case 1 % default
            switch S.GUI.TrainingLevel
                case 1 % Naive
                    PostPertDurExtra = 30;
                case 2 % Mid Trained 1
                    PostPertDurExtra = 20;
                case 3 % Mid Trained 2
                    PostPertDurExtra = 1;
                case 4 % Trained
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
        obj, S, EnableOpto)
    if S.GUI.VisStimEnable && S.GUI.AudioStimEnable
        StimAct = {};
    elseif S.GUI.VisStimEnable
        StimAct = {'HiFi1', ['P' 1]};
    elseif S.GUI.AudioStimEnable
        StimAct = {};
    else
        StimAct = {'HiFi1', ['P' 1]};
    end
    if EnableOpto
        StimAct = {};
    end
end

    end
end        