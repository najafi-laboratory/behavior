
global BpodSystem

BpodSystem.PluginObjects.S = BpodStepperModule('COM10');
BpodSystem.PluginObjects.R = RotaryEncoderModule('COM9');


%% Setup stepper module
%BpodSystem.PluginObjects.S.holdRMScurrent = 0; % set 'zero' initial lever
BpodSystem.PluginObjects.S.MaxSpeed = 1000; % set max speed
BpodSystem.PluginObjects.S.Acceleration = 1; % set acceleration
BpodSystem.PluginObjects.S.RMScurrent = 100; % set RMS current
BpodSystem.PluginObjects.S.holdRMScurrent = 0; % Immobilize lever

%% Setup rotary encoder module
BpodSystem.PluginObjects.R.setPosition(0); % Set the current position to equal 0
LastKnownEncPos = 0; % last known location of encoder pos
BpodSystem.PluginObjects.R.thresholds = 2;
BpodSystem.PluginObjects.R.sendThresholdEvents = 'on'; % If on, rotary encoder module will send threshold events to state machine
BpodSystem.PluginObjects.R.startUSBStream;
%BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_Joystick';

%% joystick trial-specific
BpodSystem.PluginObjects.R.stopUSBStream;   % stop USB streaming to update encoder params
pause(.05);
BpodSystem.PluginObjects.R.thresholds = 2;    % udate threshold from GUI params
BpodSystem.PluginObjects.R.setPosition(0);
BpodSystem.PluginObjects.R
BpodSystem.PluginObjects.R.startUSBStream;  % restart encoder USB streaming


BpodSystem.PluginObjects.S.holdRMScurrent = 30;

% BpodSystem.PluginObjects.S.holdRMScurrent = 30;
% BpodSystem.PluginObjects.S.holdRMScurrent = 900; % 
%BpodSystem.PluginObjects.S.holdRMScurrent = 0; % 
% 
Tolerance = 0.2; % Lever is home if within this Tolerance of 0, unit = degrees

%currentPosition = 999;
currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
% if (currentPosition == 999)
%     %currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
%     currentPosition = LastKnownEncPos;
%     %BpodSystem.PluginObjects.S.microStep(-5);
% end       
ramp = 0; % stepping increment to ramp the reset speed
%disp(['S.GUI.ZeroRTrials = ' num2str(S.GUI.ZeroRTrials)]);
%disp(['S.GUI.currentTrial = ' num2str(S.GUI.currentTrial)]);
% while abs(currentPosition) > Tolerance
%     %BpodSystem.PluginObjects.S.microStep(-1*degrees2MotorSteps(currentPosition, 51200)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
%     % BpodSystem.PluginObjects.S.microStep(-1 - ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
%     % ramp = ramp + 1;
%     currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
%     disp(['pos = ' num2str(currentPosition)]);
%     ramp = ramp + 1;     
%     microSteps = 50;
%     if currentPosition > 0
%         BpodSystem.PluginObjects.S.microStep(-microSteps); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step                
%     else
%         BpodSystem.PluginObjects.S.microStep(microSteps); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
%     end
%     %pause(.01); 
% end

previousPos = currentPosition;
ramp = 0;
microSteps = 100;
disp(['beh bove pos = ' num2str(currentPosition)]);
setBreakingCurrentFlag = 0;
while abs(currentPosition) > Tolerance
    
    
    if ~setBreakingCurrentFlag
        if (abs(currentPosition) - Tolerance) < 0.5
            %BpodSystem.PluginObjects.S.holdRMScurrent = 30;
            %BpodSystem.PluginObjects.S.holdRMScurrent = 900;
            BpodSystem.PluginObjects.S.MaxSpeed = 1; % set max speed
            microSteps = 10;
            setBreakingCurrentFlag = 1;
        end
    end
    %BpodSystem.PluginObjects.S.microStep(-1*degrees2MotorSteps(currentPosition, 51200)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
    % BpodSystem.PluginObjects.S.microStep(-1 - ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
    % ramp = ramp + 1;
    currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
    %disp(['pos = ' num2str(currentPosition)]);
    
    %microSteps = 1;
    % if currentPosition == previousPos
    %     %ramp = ramp + 1; % ramp
    % end
    if currentPosition > 0
        BpodSystem.PluginObjects.S.microStep(-(microSteps + ramp)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step                
    elseif currentPosition < 0
        BpodSystem.PluginObjects.S.microStep(microSteps + ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
    else
        break
    end
    % previousPos = currentPosition;
    %pause(.03);
end
BpodSystem.PluginObjects.S.MaxSpeed = 1000; % set max speed
disp(['beh low pos = ' num2str(currentPosition)]);
disp(['ramp = ' num2str(ramp)]);
currentPosition

%BpodSystem.Data.TrialData{1, S.GUI.currentTrial}.LeverResetPos = [BpodSystem.Data.TrialData{1, S.GUI.currentTrial}.LeverResetPos currentPosition]; % store position to measure start pos for mouse press
%BpodSystem.PluginObjects.S.holdRMScurrent = 30; % set lever break
BpodSystem.PluginObjects.S





% steps = 0;
% microSteps = 10;
% while abs(currentPosition) > Tolerance
%     if steps == 10
%         BpodSystem.PluginObjects.S.holdRMScurrent = 30;
%     end
%     currentPosition = BpodSystem.PluginObjects.R.currentPosition
%     BpodSystem.PluginObjects.S.microStep(-10);
%     steps = steps + 1;
% end
BpodSystem.PluginObjects.S.holdRMScurrent = 0;


BpodSystem.PluginObjects.S
BpodSystem.PluginObjects.R

pause(0.3);
BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
BpodSystem.PluginObjects.S = [];
BpodSystem.PluginObjects.R = [];
% BpodSystem.PluginObjects.S.microStep(100);
% BpodSystem.PluginObjects.S.holdRMScurrent = 30;
% BpodSystem.PluginObjects.S.microStep(100);


BpodSystem.PluginObjects.S.holdRMScurrent = 0;
ramp = 0;
microSteps = 100;
currentPosition = BpodSystem.PluginObjects.R.currentPosition;
disp(['beh bove pos = ' num2str(currentPosition)]);
while abs(currentPosition) > Tolerance
    currentPosition = BpodSystem.PluginObjects.R.currentPosition % Read the encoder
    if currentPosition > 0
        BpodSystem.PluginObjects.S.microStep(-(microSteps + ramp)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step                
    elseif currentPosition < 0
        BpodSystem.PluginObjects.S.microStep(microSteps + ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
    else
        break
    end
end
disp(['beh low pos = ' num2str(currentPosition)]);
disp(['ramp = ' num2str(ramp)]);
BpodSystem.PluginObjects.S.holdRMScurrent = 300;


currentPosition = BpodSystem.PluginObjects.R.currentPosition

% steps = 0;
% while steps < 3000
%     currentPosition = BpodSystem.PluginObjects.R.currentPosition
%     BpodSystem.PluginObjects.S.microStep(1);
%     steps = steps + 1;
% end
% 
% 
% 
% 
% BpodSystem.PluginObjects.S.microStep(100);
% BpodSystem.PluginObjects.S.holdRMScurrent = 30;
% BpodSystem.PluginObjects.S.microStep(100);