%{ 
To exit the visual stimulus presentation: hold the escape key.

Windows installation:

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

clear all;

% for ASUS VG248QG using Allen Brain Lab monitor displacement
% monitor is: 
% 53.126 cm wide
% 29.889 cm high
% 118.6 mm lateral, 86.2 mm anterior and 31.6 mm dorsal to the right eye
MonitorAnteriorDisplacement_cm = 8.62;
MonitorDorsalDisplacement_cm = 3.16;
w = 53.136;  % width of screen, in cm
h = 29.889;  % height of screen, in cm
cx = w/2 + MonitorAnteriorDisplacement_cm; % eye x location, in cm
%cy = h/2 + MonitorDorsalDisplacement_cm; % eye y location, in cm
% since we're trying to place center of monitor at center of mouse
% field of view, then these x&y should be monitor center

cx = w/2;   % eye x location, in cm
cy = h/2; % eye y location, in cm

% w = 56.69;  % width of screen, in cm
% h = 34.29;  % height of screen, in cm
% cx = w/2;   % eye x location, in cm
% %cy = 11.42; % eye y location, in cm
% cy = h/2; % eye y location, in cm

% Distance to bottom of screen, along the horizontal eye line
%zdistBottom = 24.49;     % in cm
zdistBottom = 18.337;     % in cm
%zdistTop    = 14.18;     % in cm

% Alternatively, you can specify the angle of the screen
%screenAngle = 72.5;   % in degrees, measured from table surface in front of screen to plane of screen
screenAngle = 90;   % in degrees, measured from table surface in front of screen to plane of screen
zdistTop = zdistBottom - (h*sin(deg2rad(90-screenAngle)));

% pxXmax = 200; % number of pixels in an image that fills the whole screen, x
% pxYmax = 150; % number of pixels in an image that fills the whole screen, y
pxXmax = 1920; % number of pixels in an image that fills the whole screen, x
pxYmax = 1080; % number of pixels in an image that fills the whole screen, y

% Internal conversions
top = h-cy;
bottom = -cy;
right = cx;
left = cx - w;

% Convert Cartesian to spherical coord
% In image space, x and y are width and height of monitor and z is the
% distance from the eye. I want Theta to correspond to azimuth and Phi to
% correspond to elevation, but these are measured from the x-axis and x-y
% plane, respectively. So I need to exchange the axes this way, prior to
% converting to spherical coordinates:
% orig (image) -> for conversion to spherical coords
% Z -> X
% X -> Y
% Y -> Z

[xi,yi] = meshgrid(1:pxXmax,1:pxYmax);
cart_pointsX = left + (w/pxXmax).*xi;
cart_pointsY = top - (h/pxYmax).*yi;
cart_pointsZ = zdistTop + ((zdistBottom-zdistTop)/pxYmax).*yi;
[sphr_pointsTh sphr_pointsPh sphr_pointsR] ...
            = cart2sph(cart_pointsZ,cart_pointsX,cart_pointsY);

%% view results
figure
subplot(3,2,1)
imagesc(cart_pointsX)
colorbar
title('image/cart coords, x')
subplot(3,2,3)
imagesc(cart_pointsY)
colorbar
title('image/cart coords, y')
subplot(3,2,5)
imagesc(cart_pointsZ)
colorbar
title('image/cart coords, z')

subplot(3,2,2)
imagesc(rad2deg(sphr_pointsTh))
colorbar
title('mouse/sph coords, theta')
subplot(3,2,4)
imagesc(rad2deg(sphr_pointsPh))
colorbar
title('mouse/sph coords, phi')
subplot(3,2,6)
imagesc(sphr_pointsR)
colorbar
title('mouse/sph coords, radius')


%% create grating 
Ysize = 1080;
Xsize = 1920;

% compute grating according to square grid of largest pixel dimension
if Ysize > Xsize
    gratingSize = [Ysize, Ysize]; % Size of grating in pixels
else
    gratingSize = [Xsize, Xsize]; % Size of grating in pixels
end



% Define the timing parameters
gratingDuration = .25; % Duration of grating stimulus in seconds
blankDurationOrig = 0; % Duration of gray screen stimulus in seconds
sessionduration = 10 ; % session duration in minutes

Initialcycles = 0; %50; % Number of cycles that will have blankdurationOrig only it won't be part of randsample function

oddballfrequency = 0 ; % Frequency of different blank duration
Randomdurationmin = 30;% Time in ms % Oddball min and max interval duration; uniform distribution.
Randomdurationmax = 3000; % Time in ms


% Define the parameters of the grating
%gratingSize = [1920, 1920]; % Size of grating in pixels
spatialFreq = .01; % Spatial frequency of grating in cycles per pixel
orientation = 0; % Orientation of grating in degrees
contrast = 1; % Contrast of grating (0 to 1)
phase = 0.5;
% temporalFreq = 0; % Temporal frequency of grating in Hz

FullScreen = true; % Choosing screen size
circle = false; % If true grating will be in circular shape

if FullScreen == true
    circle = false ;
end


nCycles = sessionduration*60/(blankDurationOrig + gratingDuration); % Number of cycles to present


% Calculate the parameters needed for the grating
pixPerCycle = 1 / spatialFreq;
freqPerPixel = 1 / pixPerCycle;
% cyclesPerFrame = temporalFreq / Screen('NominalFrameRate', win);
% phaseStep = cyclesPerFrame * 2 * pi * pixPerCycle;

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
sinGrating = gray + contrast/2 .* sin(freqPerPixel * 2 * pi * (cos(orientation * pi / 180) * x + sin(orientation * pi / 180) * y) + phase);
sinGrating(sinGrating > 1) = 1; % Cap values above 1 to 1 (white)
sinGrating(sinGrating < 0) = 0; % Cap values below 0 to 0 (black)

gray = gray(1:Ysize, 1:Xsize); % clip to monitor
sinGrating = sinGrating(1:Ysize, 1:Xsize); % clip to monitor

% Rescale the Cartesian maps into dimensions of radians
xmaxRad = max(sphr_pointsTh(:));
ymaxRad = max(sphr_pointsPh(:));

fx = xmaxRad/max(cart_pointsX(:));
fy = ymaxRad/max(cart_pointsY(:));

I = sinGrating;

% Apply the distortion via interpolation
ZI = interp2(cart_pointsX.*fx,cart_pointsY.*fy,I,sphr_pointsTh,sphr_pointsPh);

h=figure;
subplot(1,2,1)
imshow(I)
subplot(1,2,2)
imshow(ZI)


% Initialize Psychtoolbox
PsychDefaultSetup(2);
% Screen('Preference', 'SkipSyncTests', 1); % Skip sync tests for now
Screen('Preference', 'SkipSyncTests', 1); % Skip sync tests for now

% Open a window
% [win, winRect] = PsychImaging('OpenWindow', 0, 0);
[win, winRect] = PsychImaging('OpenWindow', 2, 0); % second argument is Monitor ID
% if above line returns incorrect screen index, can check detected monitors with Screen('Screens')

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
        % phase = phase + phaseStep;
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
            Screen('DrawTexture', win, grayTexture, [], winRect);
            
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
