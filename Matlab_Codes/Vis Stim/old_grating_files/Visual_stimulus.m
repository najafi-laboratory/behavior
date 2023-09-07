% Define the stimulus parameters
freq = 2; % Spatial frequency of the grating in cycles per degree
orientation = 90; % Orientation of the grating in degrees
contrast = 3; % Contrast of the grating
pixelsPerDegree = 100; % Pixels per degree of visual angle
phaseStep = 0.1; % Phase shift per frame in degrees

% Define the timing parameters
stimDuration = 5; % Duration of the grating stimulus in seconds
grayDuration = 5; % Duration of the gray screen stimulus in seconds
nCycles = 15; % Total number of cycles to present
changeInterval = 5; % Interval at which to change gray screen duration
minGrayDuration = 2; % Minimum duration of gray screen stimulus interval
maxGrayDuration = 10; % Maximum duration of gray screen stimulus interval

% Calculate the cycle duration
cycleDuration = stimDuration + grayDuration;

% Initialize Psychtoolbox
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1);
screens = Screen('Screens');
screenNumber = max(screens);
white = WhiteIndex(screenNumber);
black = BlackIndex(screenNumber);
[win, winRect] = PsychImaging('OpenWindow', screenNumber, black);

% Calculate the grating parameters
freqPerPixel = freq / pixelsPerDegree;
phase = 0;
inc = contrast * (white - black);

% Calculate the grating stimulus
[x, y] = meshgrid(-winRect(4)/2:winRect(4)/2-1, -winRect(3)/2:winRect(3)/2-1);
sinGrating = inc .* sin(freqPerPixel * 2 * pi * (cos(orientation * pi / 180) * x + sin(orientation * pi / 180) * y) + phase);
gratingTexture = Screen('MakeTexture', win, sinGrating);

% Calculate the gray screen stimulus
gray = ones(winRect(4), winRect(3)) * (white / 2);
grayTexture = Screen('MakeTexture', win, gray);

% Start the presentation
for iCycle = 1:nCycles
    % Check if it's time to change the gray screen duration
    if mod(iCycle-1, changeInterval) == 0 && iCycle ~= 1
        grayDuration = 5;
        cycleDuration = stimDuration + grayDuration;
    end
    
    % Present the grating stimulus
    tStart = GetSecs;
    tEnd = tStart + stimDuration;
    while GetSecs < tEnd
        phase = phase + phaseStep;
        sinGrating = inc .* sin(freqPerPixel * 2 * pi * (cos(orientation * pi / 180) * x + sin(orientation * pi / 180) * y) + phase);
        Screen('DrawTexture', win, gratingTexture, [], [], orientation);
        Screen('Flip', win);
    end
    
    % Present the gray screen stimulus
    tStart = GetSecs;
    tEnd = tStart + grayDuration;
    while GetSecs < tEnd
        Screen('DrawTexture', win, grayTexture);
        Screen('Flip', win);
    end
end

% Clean up
Screen('CloseAll');
