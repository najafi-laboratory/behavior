classdef AVstimConfig
    methods


%% Define basic video stimuli


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
    gray = gray(1:Ysize, 1:Xsize);
    gray = gray * 255;
    sinGrating = sinGrating(1:Ysize, 1:Xsize);
    sinGrating = sinGrating * 255;
    VideoGrating = repmat(sinGrating, 1, 1, 2);
    VideoGrayFixed = repmat(gray, 1, 1, 2);
end


function [Frames] = GetFrames( ...
        obj, FPS, FrameDur)
    Frames = convergent(FPS * FrameDur);
    if (mod(Frames, 2) ~= 0)
        Frames = Frames + 1;
    end
end


function [UnitVideo] = GetUnitVideo( ...
        obj, Frame_Sync, Frames)
    UnitVideo = [repmat([Frame_Sync Frame_Sync], 1, Frames/2)];
end


function [VideoData] = GetVideoDataPre( ...
        obj, S, GratingVideo, GrayVideo, RandomISI)
    if (RandomISI == 0)
        VideoData = [repmat([GratingVideo GrayVideo], 1, S.GUI.NumISIOrigRep) GratingVideo];
    else
        VideoData = [repmat([GratingVideo GrayVideo], 1, S.GUI.NumISIOrigRep) GratingVideo];
end


function [PreFrames] = GetPreFrames( ...
        obj, S, VisStim)
    PreFrames = S.GUI.NumISIOrigRep * (VisStim.Grating.Frames + VisStim.GrayPre.Frames);
end


function [PostFrames] = GetPostFrames( ...
        obj, S, VisStim)
    PostFrames = (S.GUI.NumISIOrigRep * (VisStim.Grating.Frames + VisStim.GrayPre.Frames)) ...
        * S.GUI.PostPerturbDurMultiplier;
end


function [TotalFrames] = GetTotalFrames( ...
        obj, S, VisStim)
    [PreFrames] = GetPreFrames(obj, S, VisStim);
    [PostFrames] = GetPostFrames(obj, S, VisStim);
    TotalFrames = PreFrames + PostFrames + VisStim.Grating.Frames;
end


function [FillerFrames] = GetFillerFrames(...
        obj, TotalFrames, VisStim, PertVideo, NumPerturbReps)
    FillerFrames = TotalFrames - (length(VisStim.Data.Pre) + (length(PertVideo)*NumPerturbReps));
    FillerFrames = round(FillerFrames);
    if (mod(FillerFrames, 2) ~= 0)
        FillerFrames = FillerFrames + 1;
    end
    if FillerFrames < 0
        FillerFrames = 0;
    end
end

%% Define audio stimuli


function ConfigHifi( ...
        obj, H, S, SF, Envelope)
    [InitCueSound] = GenInitCue(obj, S, SF, Envelope);
    [GoCueSound] = GenGoCue(obj, S, SF, Envelope);
    [IncorrectSound] = GenIncorrectSound(obj, S, SF, Envelope);
    H.load(1, InitCueSound);
    H.load(2, GoCueSound);
    H.load(3, IncorrectSound);
end


% Sampling freq (hz), Sine frequency (hz), duration (s)
function [InitCueSound] = GenInitCue( ...
        obj, S, SF, Envelope)
    % Sampling freq (hz), Sine frequency (hz), duration (s)
    InitCueSound = GenerateSineWave( ...
        SF, S.GUI.InitCueFreq_Hz, S.GUI.InitCueDuration_s)*S.GUI.InitCueVolume_percent;
    InitCueSound = ApplySoundEnvelope(obj, InitCueSound, Envelope);
end


% Sampling freq (hz), Sine frequency (hz), duration (s)
function [GoCueSound] = GenGoCue( ...
        obj, S, SF, Envelope)
    GoCueSound = GenerateSineWave( ...
        SF, S.GUI.GoCueFreq_Hz, S.GUI.GoCueDuration_s)*S.GUI.GoCueVolume_percent;
    GoCueSound = ApplySoundEnvelope(obj, GoCueSound, Envelope);
end


% white noise punish sound
function [IncorrectSound] = GenIncorrectSound( ...
        obj, S, SF, Envelope)
    IncorrectSound = GenerateWhiteNoise( ...
        SF, S.GUI.PunishSoundDuration_s, 1, 1)*S.GUI.IncorrectSoundVolume_percent; 
    IncorrectSound = ApplySoundEnvelope(obj, IncorrectSound, Envelope);
end


% Sampling freq (hz), Sine frequency (hz), duration (s)
function [AudioStimSound] = GenAudioStim( ...
        obj, S, SF, Envelope)
    AudioStimSound = GenerateSineWave( ...
        SF, S.GUI.AudioStimFreq_Hz, S.GUI.GratingDur_s)*S.GUI.AudioStimVolume_percent;
    AudioStimSound = ApplySoundEnvelope(obj, AudioStimSound, Envelope);
end



%% generate full envelope for sound


function [SoundWithEnvelope] = ApplySoundEnvelope( ...
        obj, Sound, Envelope)
    BackOfTheEnvelope = fliplr(Envelope);   % flipe front envelope to get back envelope
    IdxsBetweenTheEnvelope = length(Sound) - 2 * length(Envelope); % indices between front and back of envelope
    FullEnvelope = [Envelope ones(1, IdxsBetweenTheEnvelope) BackOfTheEnvelope];  % full envelope
    SoundWithEnvelope = Sound .* FullEnvelope;    % apply envelope element-wise
end


%% build audio stim to match grating/gray pattern


function [FullAudioStimData] = GenBasicAudioStim( ...
        obj, S, GratingDur, GrayDur, SF, FPS, NumPerturbReps, ...
        AudioStimSound, GrayPerturbVideo)
    PrePerturbNoSoundOffset = 1102;
    PrePerturbNoSoundOffset = 0;
    PostPerturbNoSoundOffset = 2205;
    PostPerturbNoSoundOffset = 0;
    GratingNumSamples = GratingDur * SF;     
    GrayNumSamples = GrayDur * SF;  % get duration of gray in number of audio samples for period between audio stim
    NoSoundPrePerturb = zeros(1, GrayNumSamples+PrePerturbNoSoundOffset);
    AudioPrePerturbPattern = [AudioStimSound NoSoundPrePerturb];
    PrePerturbAudioData = [repmat(AudioPrePerturbPattern, 1, S.GUI.NumISIOrigRep) AudioStimSound]; % construct preperturb audio
    GrayPerturbDur = length(GrayPerturbVideo) * (1/FPS);  % get duration of gray perturb
    GrayPerturbNumSamples = floor(GrayPerturbDur * SF);    % get duration of gray perturb in number of audio samples for period of grating
    NoSoundPerturb = zeros(1, GrayPerturbNumSamples+PostPerturbNoSoundOffset);
    AudioPerturbBasePattern = [NoSoundPerturb AudioStimSound];
    FullAudioStimData = [PrePerturbAudioData repmat(AudioPerturbBasePattern, 1, NumPerturbReps)];
end


%% Define trial-specific video stim


function [NumExtraStim] = GetNumExtraStim( ...
        obj, S)
    switch S.GUI.ExtraStim
        case 1 % default
            switch S.GUI.TrainingLevel
                case 1 % passive
                    NumExtraStim = 0;
                case 2 % Habituation
                    NumExtraStim = S.GUI.NumExtraStim;
                case 3 % Naive
                    NumExtraStim = S.GUI.NumExtraStim;
                case 4 % Mid Trained 1
                    NumExtraStim = S.GUI.NumExtraStim;
                case 5 % Mid Trained 2
                    NumExtraStim = S.GUI.NumExtraStim;
                case 6 % Trained
                    NumExtraStim = 0;
            end
        case 2 % manually extra
            NumExtraStim = S.GUI.NumExtraStim;
        case 3 % manually no extra
            NumExtraStim = 0;
    end
end


function [FullVideo] = GenFullVideo( ...
        obj, S, VisStim, PertFramesRes, GrayBlank)
    [NumExtraStim] = GetNumExtraStim(obj, S);
    if (NumExtraStim > 0)
        FullVideo = [VisStim.Data.Pre VisStim.Data.Post PertFramesRes GrayBlank];
    else
        FullVideo = [VisStim.Data.Pre VisStim.Data.Post VisStim.Filler.Video];
    end
end


function [FullAudioStimData] = ExtendedAudioStim( ...
        obj, FullAudioStimData, ...
        SF, VideoStartToGoCueFrames, FPS, ...
        GoCueSound)
    VideoStartToGoCueDur_s = VideoStartToGoCueFrames * (1/FPS);
    VideoStartToGoCueDur_NumAudioSamples = SF * VideoStartToGoCueDur_s;
    ShiftedGoCue = [zeros(1, VideoStartToGoCueDur_NumAudioSamples) GoCueSound zeros(1, length(FullAudioStimData) - VideoStartToGoCueDur_NumAudioSamples - length(GoCueSound))];
    FullAudioStimData = FullAudioStimData + ShiftedGoCue;
end


function [FullAudioStimData] = ConfigFullAudioStim( ...
        obj, H, S, GratingDur, GrayDur, SF, VideoStartToGoCueFrames, FPS, NumPerturbReps, ...
        AudioStimSound, GrayPerturbVideo, Envelope)
    [FullAudioStimData] = GenBasicAudioStim( ...
        obj, S, GratingDur, GrayDur, SF, FPS, NumPerturbReps, ...
        AudioStimSound, GrayPerturbVideo);
    [GoCueSound] = GenGoCue(obj, S, SF, Envelope);
    [NumExtraStim] = GetNumExtraStim(obj, S);
    if (NumExtraStim > 0)
        [FullAudioStimData] = ExtendedAudioStim( ...
            obj, FullAudioStimData, ...
            SF, VideoStartToGoCueFrames, FPS, ...
            GoCueSound);
    end
    H.load(5, FullAudioStimData);
end



%% enable vis/aud stim


function [OutputActionArgGoCue] = GetOutputAct( ...
        obj, S)
    if S.GUI.VisStimEnable && S.GUI.AudioStimEnable
        %OutputActionArgGoCue = {'BNCState', 1, 'BNC2', 0};
        OutputActionArgGoCue = {'BNC2', 1};
    elseif S.GUI.VisStimEnable
        %OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};
        OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNC2', 1};
    elseif S.GUI.AudioStimEnable
        %OutputActionArgGoCue = {'BNCState', 1};
        OutputActionArgGoCue = {'BNC2', 1};
    else
        %OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNCState', 1};
        OutputActionArgGoCue = {'HiFi1', ['P' 1], 'BNC2', 1};
    end
end


































    end
end        