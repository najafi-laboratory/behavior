function SoftCodeHandler(code)
    global BpodSystem
    global S
    switch S.GUI.EnableMovingSpouts
        case 1 % active moving spouts
            global M
            switch true
                case code == 255 % stop video
                    BpodSystem.PluginObjects.V.stop;
                case code == 254 % stop video and move spouts out
                    BpodSystem.PluginObjects.V.stop;
                    M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
                    M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));        
                case code >= 0 && code <= 6
                    BpodSystem.PluginObjects.V.play(code);
                case code == 8 % move spouts out
                    M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection));
                    M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection));
                case code == 9 % move spouts in
                    M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos));
                    M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos)); 
            end
        case 0 % deactive moving spouts
            switch true
                case code == 255 % stop video
                    BpodSystem.PluginObjects.V.stop;
                case code == 254 % stop video
                    BpodSystem.PluginObjects.V.stop;     
                case code >= 0 && code <= 6 % play video
                    BpodSystem.PluginObjects.V.play(code);
            end
    end
end


function SetMotorPos = ConvertMaestroPos(MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end