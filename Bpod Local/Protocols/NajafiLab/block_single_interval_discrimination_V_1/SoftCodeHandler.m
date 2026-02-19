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
    global BpodSystem
    global AntiBiasVar
    global S
    global TargetConfig
    switch S.GUI.EnableMovingSpouts
        case 1
            global M
            switch true
                case code == 254
                    M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection), S.GUI.ServoVelocity);
                    M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection), S.GUI.ServoVelocity);
                    SendBpodSoftCode(1);  % indicate maestro cmd to move servo out sent
                case code == 9
                    LeftServoSetPosition = S.GUI.LeftServoInPos + AntiBiasVar.ServoLeftAdjust;
                    RightServoSetPosition = S.GUI.RightServoInPos + AntiBiasVar.ServoRightAdjust;
                    LeftServoIsAtSetPosition = false;
                    RightServoIsAtSetPosition = false;
                    if AntiBiasVar.MoveCorrectSpout
                        switch BpodSystem.Data.TrialTypes(end)
                            case 1
                                M.setMotor(1, ConvertMaestroPos(LeftServoSetPosition), S.GUI.ServoVelocity);
                                while ~LeftServoIsAtSetPosition
                                    LeftServoIsAtSetPosition = M.checkPosition(1,LeftServoSetPosition,3);                            
                                end                                
                            case 2
                                M.setMotor(0, ConvertMaestroPos(RightServoSetPosition), S.GUI.ServoVelocity);
                                while ~RightServoIsAtSetPosition
                                    RightServoIsAtSetPosition = M.checkPosition(0,RightServoSetPosition,3);                            
                                end                                
                        end
                    else
                        M.setMotor(1, ConvertMaestroPos(LeftServoSetPosition), S.GUI.ServoVelocity); 
                        M.setMotor(0, ConvertMaestroPos(RightServoSetPosition), S.GUI.ServoVelocity);
                        while ~LeftServoIsAtSetPosition && ~RightServoIsAtSetPosition
                            % position = M.getPosition(0);
                            % disp(position)
                            RightServoIsAtSetPosition = M.checkPosition(0,RightServoSetPosition,3);
                            % position = M.getPosition(1);
                            % disp(position)
                            LeftServoIsAtSetPosition = M.checkPosition(1,LeftServoSetPosition,3);                            
                        end
                    end                                            
                    disp(['Moving left servo to: ', num2str(LeftServoSetPosition)]);
                    disp(['Moving right servo to: ', num2str(S.GUI.RightServoInPos + AntiBiasVar.ServoRightAdjust)]);                    
                    SendBpodSoftCode(2); % indicate to bpod servos have reached set position
            end
        case 0
    end
end


function SetMotorPos = ConvertMaestroPos(MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end