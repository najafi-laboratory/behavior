classdef AVstimConfig
    methods


%% Define video stimuli





%% Define audio stimuli


function [InitCueSound, GoCueSound, IncorrectSound, AudioStimSound] = GenAllSounds( ...
        obj, H, S, SF, Envelope)
    [InitCueSound] = GenInitCue(obj, S, SF, Envelope);
    [GoCueSound] = GenGoCue(obj, S, SF, Envelope);
    [IncorrectSound] = GenIncorrectSound(obj, S, SF, Envelope);
    [AudioStimSound] = GenAudioStim(obj, S, SF, Envelope);
    H.load(1, InitCueSound);
    H.load(2, GoCueSound);
    H.load(3, IncorrectSound);
end


% Sampling freq (hz), Sine frequency (hz), duration (s)
function [InitCueSound] = GenInitCue(obj, S, SF, Envelope)
    % Sampling freq (hz), Sine frequency (hz), duration (s)
    InitCueSound = GenerateSineWave( ...
        SF, S.GUI.InitCueFreq_Hz, S.GUI.InitCueDuration_s)*S.GUI.InitCueVolume_percent;
    InitCueSound = ApplySoundEnvelope(obj, InitCueSound, Envelope);
end


% Sampling freq (hz), Sine frequency (hz), duration (s)
function [GoCueSound] = GenGoCue(obj, S, SF, Envelope)
    GoCueSound = GenerateSineWave( ...
        SF, S.GUI.GoCueFreq_Hz, S.GUI.GoCueDuration_s)*S.GUI.GoCueVolume_percent;
    GoCueSound = ApplySoundEnvelope(obj, GoCueSound, Envelope);
end


% white noise punish sound
function [IncorrectSound] = GenIncorrectSound(obj, S, SF, Envelope)
    IncorrectSound = GenerateWhiteNoise( ...
        SF, S.GUI.PunishSoundDuration_s, 1, 1)*S.GUI.IncorrectSoundVolume_percent; 
    IncorrectSound = ApplySoundEnvelope(obj, IncorrectSound, Envelope);
end


% Sampling freq (hz), Sine frequency (hz), duration (s)
function [AudioStimSound] = GenAudioStim(obj, S, SF, Envelope)
    AudioStimSound = GenerateSineWave( ...
        SF, S.GUI.AudioStimFreq_Hz, S.GUI.GratingDur_s)*S.GUI.AudioStimVolume_percent;
    AudioStimSound = ApplySoundEnvelope(obj, AudioStimSound, Envelope);
end



%% generate full envelope for sound


function [SoundWithEnvelope] = ApplySoundEnvelope(obj, Sound, Envelope)
    BackOfTheEnvelope = fliplr(Envelope);   % flipe front envelope to get back envelope
    IdxsBetweenTheEnvelope = length(Sound) - 2 * length(Envelope); % indices between front and back of envelope
    FullEnvelope = [Envelope ones(1, IdxsBetweenTheEnvelope) BackOfTheEnvelope];  % full envelope
    SoundWithEnvelope = Sound .* FullEnvelope;    % apply envelope element-wise
end


%% build audio stim to match grating/gray pattern


function [FullAudioStimData] = GenBasicAudioStim( ...
        obj, S, GratingDur, GrayDur, SF, FramesPerSecond, NumPerturbReps, ...
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
    GrayPerturbDur = length(GrayPerturbVideo) * (1/FramesPerSecond);  % get duration of gray perturb
    GrayPerturbNumSamples = floor(GrayPerturbDur * SF);    % get duration of gray perturb in number of audio samples for period of grating
    NoSoundPerturb = zeros(1, GrayPerturbNumSamples+PostPerturbNoSoundOffset);
    AudioPerturbBasePattern = [NoSoundPerturb AudioStimSound];
    FullAudioStimData = [PrePerturbAudioData repmat(AudioPerturbBasePattern, 1, NumPerturbReps)];
end


function [FullAudioStimData] = ExtendedAudioStim( ...
        obj, FullAudioStimData, ...
        SF, VideoStartToGoCueFrames, FramesPerSecond, ...
        GoCueSound)
    VideoStartToGoCueDur_s = VideoStartToGoCueFrames * (1/FramesPerSecond);
    VideoStartToGoCueDur_NumAudioSamples = SF * VideoStartToGoCueDur_s;
    ShiftedGoCue = [zeros(1, VideoStartToGoCueDur_NumAudioSamples) GoCueSound zeros(1, length(FullAudioStimData) - VideoStartToGoCueDur_NumAudioSamples - length(GoCueSound))];
    FullAudioStimData = FullAudioStimData + ShiftedGoCue;
end


function [FullAudioStimData] = GenFullAudioStim( ...
        obj, S, GratingDur, GrayDur, SF, VideoStartToGoCueFrames, FramesPerSecond, NumPerturbReps, ...
        AudioStimSound, GrayPerturbVideo, GoCueSound)
    [FullAudioStimData] = GenBasicAudioStim( ...
        obj, S, GratingDur, GrayDur, SF, FramesPerSecond, NumPerturbReps, ...
        AudioStimSound, GrayPerturbVideo);
    switch S.GUI.TrainingLevel
        case 1
            [FullAudioStimData] = ExtendedAudioStim( ...
                obj, FullAudioStimData, ...
                SF, VideoStartToGoCueFrames, FramesPerSecond, ...
                GoCueSound);
        case 2
            [FullAudioStimData] = ExtendedAudioStim( ...
                obj, FullAudioStimData, ...
                SF, VideoStartToGoCueFrames, FramesPerSecond, ...
                GoCueSound);
        case 3
            [FullAudioStimData] = ExtendedAudioStim( ...
                obj, FullAudioStimData, ...
                SF, VideoStartToGoCueFrames, FramesPerSecond, ...
                GoCueSound);
        case 4
            [FullAudioStimData] = ExtendedAudioStim( ...
                obj, FullAudioStimData, ...
                SF, VideoStartToGoCueFrames, FramesPerSecond, ...
                GoCueSound);
        case 5
    end
end







































    end
end        