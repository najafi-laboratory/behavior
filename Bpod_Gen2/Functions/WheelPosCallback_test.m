function WheelPosCallback(NewPosition)
global BpodSystem
Contrast = BpodSystem.PluginObjects.GaborParams.GaborContrast; % Percent
Tilt = BpodSystem.PluginObjects.GaborParams.GaborTilt; % Degrees
Gain = BpodSystem.PluginObjects.GaborParams.GaborOffsetGain; % Factor mapping Position Offset --> Pixels from center
BpodSystem.PluginObjects.Gabor.draw(Contrast, Tilt, NewPosition*Gain)