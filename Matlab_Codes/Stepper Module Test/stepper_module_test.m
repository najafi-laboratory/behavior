global BpodSystem

B = BpodStepperModule('COM5'); % Replace 'COM3' with the actual port of the stepper module 
B.holdRMScurrent = 0;
%B.holdRMScurrent = 30; % 30mA is the first current step 