classdef AVstimConfig
    methods
        function [VideoGrating, VideoGray] = GenStimImg(~, ImgParams, Xsize, Ysize)
            imageSize = max([Xsize, Ysize]);
            [x, y] = meshgrid(1:imageSize, 1:imageSize);

            gray = 0.5 * ones(imageSize);
            grating = gray + ImgParams.contrast / 2 .* sin( ...
                2 * pi * ImgParams.spatialFreq * ...
                (cosd(ImgParams.orientation) * x + sind(ImgParams.orientation) * y) + ImgParams.phase);
            grating = min(max(grating, 0), 1);

            VideoGray = repmat(gray(1:Ysize, 1:Xsize) * 255, 1, 1, 2);
            VideoGrating = repmat(grating(1:Ysize, 1:Xsize) * 255, 1, 1, 2);
        end

        function Frames = GetFrames(~, FPS, Dur)
            Frames = 2 * ceil((FPS * Dur) / 2);
        end

        function VideoDur = GetVideoDur(~, FPS, Video)
            VideoDur = numel(Video) / FPS;
        end
    end
end
