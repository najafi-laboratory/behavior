% from https://labrigger.com/blog/2012/03/06/mouse-visual-stim/
% applies spherical warp to correct for mouse eye field of view


clear all

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
cy = h/2 + MonitorDorsalDisplacement_cm; % eye y location, in cm

% cx = w/2;   % eye x location, in cm
% cy = h/2; % eye y location, in cm

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

% view results
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

%% try a distortion

% make source image
checkSize = 20; % pixels per side of each check
% w = 100; % width, in pixels
% h = 75; % height, in pixels
w = 1920/2; % width, in pixels
h = 1080/2; % height, in pixels
I = double(checkerboard(checkSize,round(h/checkSize),round(w/checkSize))>0.5);

% alternate source image
%I = zeros(150*4,200*4);
%I(105*4:125*4,:)=0.2;
%I(20*4:40*4,:)=0.4;

% Rescale the Cartesian maps into dimensions of radians
xmaxRad = max(sphr_pointsTh(:));
ymaxRad = max(sphr_pointsPh(:));

fx = xmaxRad/max(cart_pointsX(:));
fy = ymaxRad/max(cart_pointsY(:));

% Apply the distortion via interpolation
ZI = interp2(cart_pointsX.*fx,cart_pointsY.*fy,I,sphr_pointsTh,sphr_pointsPh);

h=figure;
subplot(1,2,1)
imshow(I)
subplot(1,2,2)
imshow(ZI)


%% Here’s the line of code to use for the reverse transformation. 
% To see what the visual stimulus would look like from the mouse’s point-of-view (MPOV) if it were not corrected.
%ZI_origMPOV = griddata(sphr_pointsTh,sphr_pointsPh,I,cart_pointsX.*fx,cart_pointsY.*fy);