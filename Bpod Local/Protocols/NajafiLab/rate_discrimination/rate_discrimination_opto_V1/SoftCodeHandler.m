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
        case code >= 0 && code <= 253 % play video
            BpodSystem.PluginObjects.V.play(code);
    end
end


function HandleMoveSpouts(code)
    global S
    switch S.GUI.EnableMovingSpouts
        case 1 % active moving spouts
            global M
            switch true
                case code == 254
                    M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
                    M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));        
                case code == 8
                    M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
                    M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));
                case code == 9
                    M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos));
                    M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos)); 
            end
        case 0 % deactive moving spouts
    end
end


function SetMotorPos = ConvertMaestroPos(MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end