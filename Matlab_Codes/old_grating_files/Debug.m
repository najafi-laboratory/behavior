% Initialize Psychtoolbox
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1); % Skip sync tests for now

% Open a window
[win, winRect] = PsychImaging('OpenWindow', 0, 0);

% Define the parameters of the grating
gratingSize = [500, 500]; % Size of grating in pixels
spatialFreq = 0.04; % Spatial frequency of grating in cycles per pixel
orientation = 45; % Orientation of grating in degrees
contrast = 0.8; % Contrast of grating (0 to 1)
temporalFreq = 2; % Temporal frequency of grating in Hz
FullScreen = false; % Choosing screen size
oddballfrequency = 0.2 ; % Frequency of different blank duration

% Calculate the parameters needed for the grating
pixPerCycle = 1 / spatialFreq;
freqPerPixel = 1 / pixPerCycle;
cyclesPerFrame = temporalFreq / Screen('NominalFrameRate', win);
phaseStep = cyclesPerFrame * 2 * pi * pixPerCycle;

% Set up the grating texture
[x, y] = meshgrid(1:gratingSize(1), 1:gratingSize(2));
gray = 0.5 * ones(gratingSize);
%phase = 0.75;
phase = 0.75;
sinGrating = gray + contrast/2 .* sin(freqPerPixel * 2 * pi * (cos(orientation * pi / 180) * x + sin(orientation * pi / 180) * y) + phase);
sinGrating(sinGrating > 1) = 1; % Cap values above 1 to 1 (white)
sinGrating(sinGrating < 0) = 0; % Cap values below 0 to 0 (black)
gratingTexture = Screen('MakeTexture', win, sinGrating);

% Set up the gray texture
grayTexture = Screen('MakeTexture', win, gray);

% Define the timing parameters
gratingDuration = 2; % Duration of grating stimulus in seconds
blankDuration = 5; % Duration of gray screen stimulus in seconds
nCycles = 10; % Number of cycles to present

% Calculate the number of cycles that will have a different gray duration
nDifferentGrayDurations = round(nCycles * oddballfrequency);

% Generate a vector indicating which cycles will have a different gray duration
differentGrayDurationsIdx = randsample(nCycles, nDifferentGrayDurations);


% Start the presentation
for iCycle = 1:nCycles
    %  Check for escape key
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown && keyCode(KbName('escape'))
        break;
    end
    % Check if this is the 5th cycle
    %if mod(iCycle, 5) == 0
    % Check if this cycle should have a different gray duration
    if any(iCycle == differentGrayDurationsIdx)
        % If so, change the duration of the gray screen stimulus
        blankDuration = randi([10, 20]); % convert to seconds
    else
        % Otherwise, use the default duration
        blankDuration = 0.75;
    end
    
    % Present the grating stimulus
    tStart = GetSecs;
    tEnd = tStart + gratingDuration;
    while GetSecs < tEnd
        phase = phase + phaseStep;
        sinGrating = gray + contrast/2 .* sin(freqPerPixel * 2 * pi * (cos(orientation * pi / 180) * x + sin(orientation * pi / 180) * y) + phase);
        sinGrating(sinGrating > 1) = 1;
        sinGrating(sinGrating < 0) = 0;
        %Screen('DrawTexture', win, gratingTexture, [], [], orientation);
        if FullScreen == true
            Screen('DrawTexture', win, gratingTexture, [], winRect);
            %Screen('DrawTexture', win, gratingTexture, [], winRect, [], [], [], [], [], kPsychUseTextureMatrixForRotation, [orientation, 0.5 * gratingSize(1), 0.5 * gratingSize(2)]);

        else
            Screen('DrawTexture', win, gratingTexture, [], []);
        end
        Screen('Flip', win);
    end
    
    % Present the gray screen stimulus
    tStart = GetSecs;
    tEnd = tStart + blankDuration;
    while GetSecs < tEnd
        %Screen('DrawTexture', win, grayTexture);
        if FullScreen == true
            Screen('DrawTexture', win, grayTexture, [], winRect);
            
        else
            Screen('DrawTexture', win, grayTexture,[],[]);
        end
            Screen('Flip', win);

    end
end

% Clean Up
Screen('CloseAll');
