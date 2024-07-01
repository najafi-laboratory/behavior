function SoftCodeHandler_EBC(code)
% global M
global S
global BpodSystem
global MEV

switch true
    % case code == 7
    %     % disp('code 7');        
    %     M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - S.GUI.ServoOutPos));        
    %     SendBpodSoftCode(1); % Indicate to the state machine that the horiz bar is open for press
    % case code == 8
    %     % disp('code 8');
    %     M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos), 0.5);
    %     % pause(1); % 200ms after starting retract prior to ITI or next stim/press
    %     % Tolerance = 0.4; % Lever is home if within this Tolerance of 0, unit = degrees
    %     Tolerance = S.GUI.RetractThreshold;
    % 
    %     % might need to add something here to abort if current position
    %     % isn't retreived within 500 ms or so
    %     currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the enc
    % 
    %     while abs(currentPosition) > Tolerance
    %         currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder           
    %         % disp(['pos = ' num2str(currentPosition)]);
    % 
    %     end
    %     % pause(.01);
    %     SendBpodSoftCode(1); % Indicate to the state machine that the lever is back in the home position
    case code >= 0 && code <= 6 
        BpodSystem.PluginObjects.V.play(code);
    case code == 7
        MEV.startVideoTrial;
        SendBpodSoftCode(1); % Indicate to the state machine that video start
    case code == 8
        MEV.stopVideoTrial;  
        SendBpodSoftCode(1); % Indicate to the state machine that video stop
    case code == 255
        BpodSystem.PluginObjects.V.stop;
end
end

% function steps = degrees2MotorSteps(degrees, nMotorStepsPerRev)
%     steps = round((degrees/360)*nMotorStepsPerRev);
% end
% 
% function SetMotorPos = ConvertMaestroPos(MaestroPosition)
%     m = 0.002;
%     b = -3;
%     SetMotorPos = MaestroPosition * m + b;
% end