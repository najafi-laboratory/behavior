function [video, duration] = GenerateVisualCueVideo(imagePath, width, height, fps, requestedDuration, useGeneratedGrating)
% Resize image.png or generate the sinusoidal grating cue frame.
if nargin < 6
    useGeneratedGrating = false;
end

if useGeneratedGrating
    % Build a synthetic grating when no external cue image is wanted.
    frame = generatedGrating(width, height);
else
    if ~isfile(imagePath)
        error('Visual cue image not found: %s', imagePath)
    end

    frame = imread(imagePath);
    if size(frame, 3) == 4
        frame = frame(:, :, 1:3);
    end

    % Resize by indexed sampling so the output matches the display viewport.
    rows = round(linspace(1, size(frame, 1), height));
    columns = round(linspace(1, size(frame, 2), width));
    frame = frame(rows, columns, :);

    if isfloat(frame)
        frame = uint8(255 * min(max(frame, 0), 1));
    end
end

% Snap duration to the nearest whole video frame count.
frameCount = max(1, round(fps * requestedDuration));
duration = frameCount / fps;
video = frame;
end

function frame = generatedGrating(width, height)
% Generate one grayscale sinusoidal cue frame.
imageSize = max([width height]);
[x, y] = meshgrid(1:imageSize, 1:imageSize);

spatialFreq = 0.005;
orientation = 0;
contrast = 1;
phase = 0.5;

gray = 0.5 * ones(imageSize);
grating = gray + contrast / 2 .* sin( ...
    2 * pi * spatialFreq * (cosd(orientation) * x + sind(orientation) * y) + phase);
grating = min(max(grating, 0), 1);
frame = uint8(grating(1:height, 1:width) * 255);
end
