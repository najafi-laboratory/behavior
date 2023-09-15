function WheelPosCallback(NewPosition)
global BpodSystem
disp(['NewPosition: ', num2str(NewPosition)]);
%Contrast = BpodSystem.PluginObjects.GaborParams.GaborContrast; % Percent
%Tilt = BpodSystem.PluginObjects.GaborParams.GaborTilt; % Degrees
%Gain = BpodSystem.PluginObjects.GaborParams.GaborOffsetGain; % Factor mapping Position Offset --> Pixels from center
%BpodSystem.PluginObjects.Gabor.draw(Contrast, Tilt, NewPosition*Gain)