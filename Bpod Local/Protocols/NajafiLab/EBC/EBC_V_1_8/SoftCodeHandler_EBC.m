function SoftCodeHandler_EBC(code)
% global M
global S
global BpodSystem
global MEV

switch true
    case code >= 0 && code <= 6 
        BpodSystem.PluginObjects.V.play(code);
    case code == 7
        % MEV.startVideoTrial;
        SendBpodSoftCode(1); % Indicate to the state machine that video start
    case code == 8
        % MEV.stopVideoTrial;  
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