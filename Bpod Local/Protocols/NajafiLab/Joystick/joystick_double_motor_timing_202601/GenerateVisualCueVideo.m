function [video, duration] = GenerateVisualCueVideo(imagePath, width, height, fps, requestedDuration)
% Resize image.png into the display-sized cue frame used by PsychToolbox.
if ~isfile(imagePath)
    error('Visual cue image not found: %s', imagePath)
end

frame = imread(imagePath);
if size(frame, 3) == 4
    frame = frame(:, :, 1:3);
end

rows = round(linspace(1, size(frame, 1), height));
columns = round(linspace(1, size(frame, 2), width));
frame = frame(rows, columns, :);

if isfloat(frame)
    frame = uint8(255 * min(max(frame, 0), 1));
end

frameCount = max(1, round(fps * requestedDuration));
duration = frameCount / fps;
video = frame;
end
