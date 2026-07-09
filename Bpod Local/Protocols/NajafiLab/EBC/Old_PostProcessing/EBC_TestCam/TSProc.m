% load session data
path = "C:\behavior\session_data\FakeSubject";
filename = "FakeSubject_EBC_Opto_V_5_3_20251230_141439.mat";

fullpath = fullfile(path, filename);
SessionData = load(fullpath).SessionData;



% load session video
path = "C:\behavior\video_data\FakeSubject\20251230";
filename = "FakeSubject_SessionVid_2025-12-30_141510.avi";

video_fullpath = fullfile(path, filename);
v = VideoReader(video_fullpath);


nFrames = 10358;
nPulses = 10364;

frames = zeros(v.Height, v.Width, 1, v.NumFrames, 'uint8');

Second_count = [];
Cycle_count = [];
Cycle_offset = [];
FrameNum = [];

% for k = 1:v.NumFrames
for k = 1:3000   
    frame = readFrame(v);
    frames(:,:,:,k) = frame;
    % imshow(frame);

    timechunk = frame(1, 1:4);
    % imshow(timechunk);

    % x = uint8([b1 b2 b3 b4]);   % 1×4 uint8
    x = timechunk;

    word = uint32(0);
    word = bitor(word, bitshift(uint32(x(1)), 24));
    word = bitor(word, bitshift(uint32(x(2)), 16));
    word = bitor(word, bitshift(uint32(x(3)),  8));
    word = bitor(word, uint32(x(4)));


    A = bitshift(word, -25);                    % top 7 bits
    B = bitand(bitshift(word, -12), 2^13 - 1);  % next 13 bits
    % C = bitand(word, 2^12 - 1);                 % last 12 bits     
    C = bitand(word, (2^12 - 1) - (2^4-1));     % lower 12 bits, with lower 4 zeros
    D = bitand(word, 2^4 - 1);                  % lower 4 bits


    Second_count = [Second_count A];
    Cycle_count = [Cycle_count B];
    Cycle_offset = [Cycle_offset C];
    FrameNum = [FrameNum D];
    % on USB the four LSBs do not reflect cycle_offset

    

end

% while hasFrame(v)
%     frame = readFrame(v);   % H × W × 3 uint8
% 
%     % --- your per-frame processing here ---
% end


Second_count_diff = diff(Second_count);
Cycle_count_diff = diff(Cycle_count);
Cycle_offset_diff = diff(Cycle_offset);




Cycle_count_seconds = double(Cycle_count) / 7999;
Cycle_count_seconds_diff = diff(Cycle_count_seconds);

% max(Cycle_count_seconds_diff)

timestamp = double(Second_count) + Cycle_count_seconds;
timestamp_diff = diff(timestamp);

max(timestamp_diff);

d = timestamp_diff;
n = nPulses - v.NumFrames;
[largestDiffs, idx] = maxk(d, n); 


vidTime = SessionData.vidTime;
vidTimeDiff = diff(vidTime);




local_l = idx - 20;
local_r = idx + 20;

local_median = median(vidTimeDiff(local_l(1):local_r(1)))

dropSegArray = vidTimeDiff(1970:1999)
tot_time = sum(dropSegArray)
numImgs = length(dropSegArray)


x = tot_time - (numImgs * local_median)
% x = tot_time - (numImgs * 0.0035)
num_missing_at_anomoly = x / local_median
num_missing_imgs = round(num_missing_at_anomoly)
% 
% EBC_Opto_V_5_3 ended
% Pulse count = 10364
% 
% ans =
% 
%         4594
% 
% 
% ans =
% 
%        10358
% 
% 
% n =
% 
%     23
% 
% Writing Video File...
% Elapsed time is 4.135451 seconds.
% Processing FEC Data...
% Elapsed time is 12.811659 seconds.
% FramesAcquired = 10358
% 
% trialSyncChannel_numRising =
% 
%      2
% 
% 
% camStrobeChannel_numRising =
% 
%        10364
% 
% 
% ans =
% 
%        10358
% 
% FramesAcquired = 10358
