%{ 
Windows installation
Filesystem locations given here are examples. You could choose other disc drives or folders of your liking instead:
Skip this step and go directly to step 2 if you use Matlab R2014b or later, instead of an older Matlab version or of GNU/Octave.
Download and install the Subversion installer
Windows: Subversion 1.7.x command-line client
Download the Psychtoolbox installer to your desktop.
You must install the 64-Bit GStreamer-1.20.5 MSVC or later versions runtime from gstreamer.freedesktop.org even if you do not need multi-media support! Do not install the MINGW variant, it will not work, but likely crash! Make absolutely sure that you install all offered packages. Read help GStreamer carefully for this purpose, before downloading and installing GStreamer.
If you intend to use Octave, you will need to delete the following DLL files from the C:\Program Files\GNU Octave\Octave-7.3.0\mingw64\bin\ folder: opengl32.dll – Otherwise hardware accelerated visual stimulation will not work.
You may also need to install the Microsoft Runtime Libraries for MSVC 2015-2019 if you use Matlab instead of Octave. For a few use cases you may even need those if you use Octave. You can find installers for these at Microsoft’s site beforehand. Otherwise when our installer aborted half-ways, follow the instructions it prints to the console. Or simply click this link to get a copy bundled with Psychtoolbox
Open Matlab as administrative user (right-click the Matlab shortcut and “Run As Administrator”) and type the following in the command window, assuming you want Psychtoolbox to be installed inside the C:\toolbox folder:
>> cd('into the folder where you downloaded DownloadPsychtoolbox.m to').
>> DownloadPsychtoolbox('C:\toolbox')
The second command will take a bit of time (a few minutes in some cases) and may generate a lot of output. Please be patient (and make sure your computer is not going to go onto standby while installing). You may get the command line reappear before the installation is finished - so don’t assume the command line reappearing means that installation has hung. The installer will tell you when it is finished.
%}

% Initialize Psychtoolbox
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1); % Skip sync tests for now

% Open a window
[win, winRect] = PsychImaging('OpenWindow', 0, 0);

% Define the parameters of the grating
gratingSize = [500, 500]; % Size of grating in pixels
spatialFreq = 0.01; % Spatial frequency of grating in cycles per pixel
orientation = 45; % Orientation of grating in degrees
contrast = 0.8; % Contrast of grating (0 to 1)
temporalFreq = 2; % Temporal frequency of grating in Hz
FullScreen = true; % Choosing screen size
oddballfrequency = 0.2 ; % Frequency of different blank duration
circle = false; % If true grating will be in circular shape

if FullScreen == true
    circle = false ;
end


% Define the timing parameters
gratingDuration = .25; % Duration of grating stimulus in seconds
blankDurationOrig = 1; % Duration of gray screen stimulus in seconds
sessionduration = 10 ; % session duration in minutes
nCycles = sessionduration*60/(blankDurationOrig + gratingDuration); % Number of cycles to present
Initialcycles = 50; % Number of cycles that will have blankdurationOrig only it won't be part of randsample function
Randomdurationmin = 30;% Time in ms
Randomdurationmax = 3000; % Time in ms

% Calculate the parameters needed for the grating
pixPerCycle = 1 / spatialFreq;
freqPerPixel = 1 / pixPerCycle;
cyclesPerFrame = temporalFreq / Screen('NominalFrameRate', win);
phaseStep = cyclesPerFrame * 2 * pi * pixPerCycle;

% Set up the grating texture
if circle == false 
    [x, y] = meshgrid(1:gratingSize(1), 1:gratingSize(2));
else 
    [x, y] = meshgrid(-gratingSize(1)/2:gratingSize(1)/2-1, -gratingSize(2)/2:gratingSize(2)/2-1);
    aperture = sqrt(x.^2 + y.^2) < (min(gratingSize)/2);
    x(~aperture) = nan;
    y(~aperture) = nan;
end
    gray = 0.5 * ones(gratingSize);
    phase = 0.75;
    sinGrating = gray + contrast/2 .* sin(freqPerPixel * 2 * pi * (cos(orientation * pi / 180) * x + sin(orientation * pi / 180) * y) + phase);
    sinGrating(sinGrating > 1) = 1; % Cap values above 1 to 1 (white)
    sinGrating(sinGrating < 0) = 0; % Cap values below 0 to 0 (black)
    gratingTexture = Screen('MakeTexture', win, sinGrating);



% Set up the gray texture
if circle == false
    grayTexture = Screen('MakeTexture', win, gray);
else
    gray = 0.5 * ones(gratingSize);
    [x, y] = meshgrid(-gratingSize(1)/2:gratingSize(1)/2-1, -gratingSize(2)/2:gratingSize(2)/2-1);
    grayAperture = sqrt(x.^2 + y.^2) < (min(gratingSize)/2);
    gray(~grayAperture) = nan;
    grayTexture = Screen('MakeTexture', win, gray);
end
% Calculate the number of cycles that will have a different gray duration
nDifferentGrayDurations = round((nCycles-Initialcycles) * oddballfrequency);

% Generate a vector indicating which cycles will have a different gray duration
differentGrayDurationsIdx = randsample(Initialcycles+1:nCycles, nDifferentGrayDurations);

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
        blankDuration = randi([Randomdurationmin, Randomdurationmax])/1000; % convert to seconds
    else
        % Otherwise, use the default duration
        blankDuration = blankDurationOrig;
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
        else
            Screen('DrawTexture', win, gratingTexture, [], []);
        end
        Screen('Flip', win);
        %  Check for escape key
        [keyIsDown, ~, keyCode] = KbCheck;
        if keyIsDown && keyCode(KbName('escape'))
            break;
        end
    end
    
    % Present the gray screen stimulus
    tStart = GetSecs;
    tEnd = tStart + blankDuration;
    while GetSecs < tEnd
        %Screen('DrawTexture', win, grayTexture);
        if FullScreen == true
            Screen('DrawTexture', win, grayTexture, [],winRect);
            
        else
            Screen('DrawTexture', win, grayTexture,[],[]);
        end
            Screen('Flip', win);
            %  Check for escape key
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown && keyCode(KbName('escape'))
                break;
            end

    end
end

% Clean Up
Screen('CloseAll');
