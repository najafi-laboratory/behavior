function SoftCodeHandler_LeverReset(code)
global BpodSystem

BpodSystem.PluginObjects.S.holdRMScurrent = 200; % Immobilize the lever
Tolerance = 1; % Lever is home if within this Tolerance of 0, unit = degrees
currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
while abs(currentPosition) > Tolerance
    BpodSystem.PluginObjects.S.microStep(-1*degrees2MotorSteps(currentPosition, 51200)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
    currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
    pause(.01);
end
SendBpodSoftCode(1); % Indicate to the state machine that the lever is back in the home position

function steps = degrees2MotorSteps(degrees, nMotorStepsPerRev)
steps = round((degrees/360)*nMotorStepsPerRev);