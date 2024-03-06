function SoftCodeHandler_Joystick(code)
global M
global S
global BpodSystem

switch true
    case code == 7
        disp('code 7');        
        M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - S.GUI.ServoOutPos));        
        SendBpodSoftCode(1); % Indicate to the state machine that the horiz bar is open for press
    case code == 8
        disp('code 8');
        M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos), 0.5);
        pause(0.2); % 200ms after starting retract prior to ITI or next stim/press
        SendBpodSoftCode(1); % Indicate to the state machine that the lever is back in the home position
    case code >= 0 && code <= 6 
        BpodSystem.PluginObjects.V.play(code);
    case code == 255
        BpodSystem.PluginObjects.V.stop;
end
end

function steps = degrees2MotorSteps(degrees, nMotorStepsPerRev)
    steps = round((degrees/360)*nMotorStepsPerRev);
end

function SetMotorPos = ConvertMaestroPos(MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end