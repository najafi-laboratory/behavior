% Heavy-compress a video while retaining all frames.
inputVid  = 'C:\behavior\video_data\FakeSubject\260102\FakeSubject_SessionVid_2026-01-02_141634.avi';
[folder,name,~] = fileparts(inputVid);
outputVid = fullfile(folder, [name '_compressed.mp4']);

vIn  = VideoReader(inputVid);
vOut = VideoWriter(outputVid, 'Motion JPEG AVI');
vOut.FrameRate = vIn.FrameRate;
vOut.Quality   = 3;   % 0–100, lower = smaller size; adjust as needed
open(vOut);

% estimate total frames
if ~isempty(vIn.NumFrames)
    totalFrames = vIn.NumFrames;
else
    totalFrames = round(vIn.Duration * vIn.FrameRate);
end

fprintf('Compressing to %s (est. %d frames)\n', outputVid, totalFrames);
k = 0;
reportEvery = 5000;

while hasFrame(vIn)
    frame = readFrame(vIn);
    writeVideo(vOut, frame);
    k = k + 1;
    if mod(k, reportEvery) == 0
        pct = 100 * k / totalFrames;
        fprintf('  wrote %d / %d (%.1f%%)\n', k, totalFrames, pct);
    end
end
close(vOut);
fprintf('Done. Wrote %d frames.\n', k);