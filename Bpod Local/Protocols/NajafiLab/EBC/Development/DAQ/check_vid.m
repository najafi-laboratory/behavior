vid_path = "C:\behavior\video_data\FakeSubject\251221\FakeSubject_SessionVid_2025-12-21_200247.avi";
vid = VideoReader(vid_path);  % replace with your file
numFrames = 0;

while hasFrame(vid)
    readFrame(vid);   % read and discard
    numFrames = numFrames + 1;
end

disp(numFrames)