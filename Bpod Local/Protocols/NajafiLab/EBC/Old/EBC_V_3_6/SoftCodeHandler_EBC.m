function SoftCodeHandler_EBC(code)
% global M
global S
global BpodSystem
global MEV

switch true  
    case code == 1
        % MEV.stopVideoTrial;  

        MEV.eyeOpen = false;
        while ~MEV.eyeOpen
            MEV.checkEyeOpen();
            pause(0.03);
        end

        SendBpodSoftCode(1); % Indicate to the state machine that eye is open relative to threshold   
    case code == 2
        % MEV.startVideoTrial;
        MEV.triggerTrialsVideo();
        SendBpodSoftCode(1); % Indicate to the state machine that video start
    % case code == 8
    %     % MEV.stopVideoTrial;  
    %     SendBpodSoftCode(1); % Indicate to the state machine that video stop
    % case code == 10
    %     MEV.LEDOnsetTime = datetime("now");
    %     MEV.plotLEDOnset;
    % 
    % case code == 11
    %     MEV.AirPuffOnsetTime = datetime("now");

    case code >= 20 && code <= 30
        BpodSystem.PluginObjects.V.play(code);
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