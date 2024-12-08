function SoftCodeHandler(code)
    HandleVideo(code)
    HandleMoveSpouts(code)
end


function HandleVideo(code)
    global BpodSystem
    switch true
        case code == 255 % stop video
            BpodSystem.PluginObjects.V.stop;
        case code == 254 % stop video
            BpodSystem.PluginObjects.V.stop;     
        case code == 25 % play video
            BpodSystem.PluginObjects.V.play(code);
    end
end


function HandleMoveSpouts(code)
    global AntiBiasVar
    global S
    switch S.GUI.EnableMovingSpouts
        case 1
            global M
            switch true
                case code == 254
                    M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
                    M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));        
                case code == 9
                    M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos + AntiBiasVar.ServoRightAdjust));
                    M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos + AntiBiasVar.ServoLeftAdjust)); 
                    disp(['Moving right servo to: ', num2str(S.GUI.RightServoInPos + AntiBiasVar.ServoRightAdjust)]);
                    disp(['Moving left servo to: ', num2str(S.GUI.LeftServoInPos + AntiBiasVar.ServoLeftAdjust)]);
            end
        case 0
    end
end


function SetMotorPos = ConvertMaestroPos(MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end