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


        % MEV.startVideoTrial;
        SendBpodSoftCode(1); % Indicate to the state machine that video start
    case code == 8
        % MEV.stopVideoTrial;  

        MEV.stopVideoTrial;
        SendBpodSoftCode(1); % Indicate to the state machine that video stop
    case code == 9
        % MEV.stopVideoTrial;  

        while MEV.vid.FramesAvailable > 0
            if MEV.vid.FramesAvailable == MEV.vid.FramesPerTrigger
                MEV.checkEyeOpenness;
                % if ~isrunning(MEV.vid)
                %     start(MEV.vid);
                % end
            end
        end
        if MEV.eyeOpen
            SendBpodSoftCode(2); % Indicate to the state machine that eye is open
        else
            SendBpodSoftCode(1); % Indicate to the state machine that eye is closed    
        end

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