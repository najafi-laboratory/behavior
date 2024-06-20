
FT = Frame2TTLTestTool;
FT.SyncPatchIntensity = 255; % Set this to match the sync patch intensity you use in your protocol






F = Frame2TTLv2('COM5');


FT.setPatch(0); % Set the sync patch to 'Dark'
F.setLightThreshold_Auto;

FT.setPatch(1); % Set the sync patch to 'Light'
F.setDarkThreshold_Auto;

%6. F.LightThreshold and F.DarkThreshold are the calibrated detection thresholds. In the setup section of your protocol, program them with:
F.DarkThreshold = dark_val;
F.LightThreshold = light_val;

FT.playVideo; % Play a 1s test video

% clear FT; % Close the test tool and release the video display