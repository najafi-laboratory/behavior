% Heavy-compress a video while retaining all frames.
inputVid  = 'C:\behavior\video_data\FakeSubject\260121\FakeSubject_SessionVid_2026-01-21_094825.avi';
[folder,name,~] = fileparts(inputVid);
outputVid = fullfile(folder, [name '_compressed.mp4']);
outputData = fullfile(folder, [name '_Timestamps.mat']);

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
reportEvery = 5000;


secArr = [];
cycArr = [];
offArr = [];
nibArr = [];            
frameIdx = 1;
while hasFrame(vIn)
    frame = readFrame(vIn);

    % Take first four bytes (big-endian) from the raw frame buffer
    x = frame(1, 1:4);

    word = uint32(0);
    word = bitor(word, bitshift(uint32(x(1)), 24));
    word = bitor(word, bitshift(uint32(x(2)), 16));
    word = bitor(word, bitshift(uint32(x(3)),  8));
    word = bitor(word, uint32(x(4)));

    A = bitshift(word, -25);                    % top 7 bits (seconds)
    B = bitand(bitshift(word, -12), 2^13 - 1);  % next 13 bits (cycle count)
    C = bitand(word, (2^12 - 1) - (2^4-1));     % lower 12 bits, low 4 zeroed (cycle offset)
    D = bitand(word, 2^4 - 1);                  % lower 4 bits (frame number nibble)

    secArr(frameIdx,1) = A; %#ok<AGROW>
    cycArr(frameIdx,1) = B; %#ok<AGROW>
    offArr(frameIdx,1) = C; %#ok<AGROW>
    nibArr(frameIdx,1) = D; %#ok<AGROW>

    writeVideo(vOut, frame);
    % k = k + 1;
    
    if mod(frameIdx, reportEvery) == 0
        pct = 100 * frameIdx / totalFrames;
        fprintf('  wrote %d / %d (%.1f%%)\n', frameIdx, totalFrames, pct);
    end
    frameIdx = frameIdx + 1;
end
close(vOut);
fprintf('Done. Wrote %d frames.\n', k);


unwrapNib = nibArr(:);
wrapOffset = 0;
for ii = 2:numel(unwrapNib)
    if unwrapNib(ii) < nibArr(ii-1)
        wrapOffset = wrapOffset + 16;
    end
    unwrapNib(ii) = unwrapNib(ii) + wrapOffset;
end
unwrapNib = unwrapNib - (unwrapNib(1) - 1);
frameNumArr = unwrapNib;

VidTimestamps.frameNumArr;

VidTimestamps.secondCount = secArr;
VidTimestamps.cycleCount = cycArr;
VidTimestamps.cycleOffset = offArr;
VidTimestamps.frameNibble = nibArr;

% compute timestamp (seconds + cycle/7999)
cycleSeconds = double(VidTimestamps.cycleCount) / 7999;
timestampSec = double(VidTimestamps.secondCount) + cycleSeconds;

VidTimestamps.timestampSec = timestampSec;

% SessionData = obj.SessionData;
% save(obj.sessionDataPath, '-struct', 'S');
save(outputData, 'VidTimestamps');

