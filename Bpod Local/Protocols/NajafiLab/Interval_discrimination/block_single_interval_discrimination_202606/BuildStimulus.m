function stimulus = BuildStimulus(S, isi)
% Build synchronized visual and audio stimuli for one trial.
global BpodSystem

width = BpodSystem.PluginObjects.V.ViewPortDimensions(1);
height = BpodSystem.PluginObjects.V.ViewPortDimensions(2);
fps = BpodSystem.PluginObjects.V.DetectedFrameRate;
sampleRate = BpodSystem.PluginObjects.H.SamplingRate;

gratingFrames = max(1, round(S.GUI.GratingDuration_s * fps));
isiFrames = max(1, round(isi * fps));
actualGratingDuration = gratingFrames / fps;
actualISI = isiFrames / fps;

gratingFrame = stimulusImage(width, height, S);
grayFrame = uint8(127 * ones(height, width, 3));
loadBaseFrames(gratingFrame, grayFrame);

gratingSyncWhite = BpodSystem.PluginObjects.V.Videos{21}.Data(1);
graySyncWhite = BpodSystem.PluginObjects.V.Videos{22}.Data(1);
graySyncBlack = BpodSystem.PluginObjects.V.Videos{22}.Data(2);

visualFrames = [repmat(gratingSyncWhite, 1, gratingFrames), repmat(graySyncBlack, 1, isiFrames), repmat(gratingSyncWhite, 1, gratingFrames)];
syncFrames = [repmat(graySyncWhite, 1, gratingFrames), repmat(graySyncBlack, 1, isiFrames), repmat(graySyncWhite, 1, gratingFrames)];
audio = audioStimulus(S, actualGratingDuration, actualISI, sampleRate);

if S.GUI.StimulusMode == 1
    audio = zeros(1, max(1, numel(audio)));
elseif S.GUI.StimulusMode == 2
    visualFrames = syncFrames;
end

stimulus.VisualVideo = visualFrames;
stimulus.SyncVideo = syncFrames;
stimulus.Audio = audio;
stimulus.Duration_s = 2 * actualGratingDuration + actualISI;
stimulus.GratingDuration_s = actualGratingDuration;
stimulus.ISI_s = actualISI;
stimulus.Mode = S.GUI.StimulusMode;
stimulus.UseSavedImage = S.GUI.UseSavedImage;
end

function loadBaseFrames(gratingFrame, grayFrame)
global BpodSystem

BpodSystem.PluginObjects.V.loadVideo(21, gratingFrame);
BpodSystem.PluginObjects.V.loadVideo(22, grayFrame);
end

function frame = stimulusImage(width, height, S)
if S.GUI.UseSavedImage
    frame = savedImage(width, height);
else
    frame = gratingImage(width, height);
end
end

function frame = savedImage(width, height)
imagePath = fullfile(fileparts(mfilename('fullpath')), 'image.png');
if ~isfile(imagePath)
    error('Saved stimulus image not found: %s', imagePath)
end
frame = imread(imagePath);
if size(frame, 3) == 4
    frame = frame(:, :, 1:3);
end
if isfloat(frame)
    frame = uint8(255 * min(max(frame, 0), 1));
end
if size(frame, 3) == 1
    frame = repmat(frame, 1, 1, 3);
end
rows = round(linspace(1, size(frame, 1), height));
columns = round(linspace(1, size(frame, 2), width));
frame = frame(rows, columns, :);
end

function frame = gratingImage(width, height)
imageSize = max(width, height);
[x, y] = meshgrid(1:imageSize, 1:imageSize);

orientation = 0;
spatialFreq = 0.005;
contrast = 1;
phase = 0.5;
gray = 0.5 * ones(imageSize);
grating = gray + contrast / 2 .* sin( ...
    2 * pi * spatialFreq * (cosd(orientation) * x + sind(orientation) * y) + phase);
grating = min(max(grating, 0), 1);
mono = uint8(grating(1:height, 1:width) * 255);
frame = repmat(mono, 1, 1, 3);
end

function audio = audioStimulus(S, gratingDuration, isi, sampleRate)
tone = sineTone(S.GUI.AudioStimFreq_Hz, gratingDuration, S.GUI.AudioStimVolume, sampleRate, S.GUI.AudioRamp_ms);
silence = zeros(1, max(1, round(isi * sampleRate)));
audio = [tone silence tone];
end

function tone = sineTone(frequency, duration, volume, sampleRate, rampMs)
sampleCount = max(1, round(duration * sampleRate));
t = (0:sampleCount - 1) / sampleRate;
tone = sin(2 * pi * frequency * t) * volume;
rampSamples = min(floor(sampleCount / 2), max(1, round(rampMs / 1000 * sampleRate)));
ramp = linspace(0, 1, rampSamples);
envelope = ones(1, sampleCount);
envelope(1:rampSamples) = ramp;
envelope(end - rampSamples + 1:end) = fliplr(ramp);
tone = tone .* envelope;
end
