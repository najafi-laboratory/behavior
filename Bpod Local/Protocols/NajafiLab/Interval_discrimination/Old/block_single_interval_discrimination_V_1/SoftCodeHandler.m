function SoftCodeHandler(code)
HandleVideo(code);
HandleMoveSpouts(code);
end

function HandleVideo(code)
global BpodSystem
switch code
    case {254, 255}
        BpodSystem.PluginObjects.V.stop;
    case 25
        BpodSystem.PluginObjects.V.play(code);
end
end

function HandleMoveSpouts(code)
global BpodSystem AntiBiasVar S M

if S.GUI.EnableMovingSpouts ~= 1
    return
end

trialConfig = TrialConfig;
switch code
    case 254
        M.setMotor(0, trialConfig.ConvertMaestroPos(S.GUI.RightServoInPos - S.GUI.ServoDeflection), S.GUI.ServoVelocity);
        M.setMotor(1, trialConfig.ConvertMaestroPos(S.GUI.LeftServoInPos + S.GUI.ServoDeflection), S.GUI.ServoVelocity);
        SendBpodSoftCode(1);
    case 9
        moveSpoutsToTrialPosition(trialConfig);
        SendBpodSoftCode(2);
end
end

function moveSpoutsToTrialPosition(trialConfig)
global BpodSystem AntiBiasVar S M

leftServoPos = S.GUI.LeftServoInPos + AntiBiasVar.ServoLeftAdjust;
rightServoPos = S.GUI.RightServoInPos + AntiBiasVar.ServoRightAdjust;

if AntiBiasVar.MoveCorrectSpout
    switch BpodSystem.Data.TrialTypes(end)
        case 1
            moveAndWait(M, 1, leftServoPos, S.GUI.ServoVelocity, trialConfig);
        case 2
            moveAndWait(M, 0, rightServoPos, S.GUI.ServoVelocity, trialConfig);
    end
else
    M.setMotor(1, trialConfig.ConvertMaestroPos(leftServoPos), S.GUI.ServoVelocity);
    M.setMotor(0, trialConfig.ConvertMaestroPos(rightServoPos), S.GUI.ServoVelocity);
    waitForBothSpouts(M, leftServoPos, rightServoPos);
end

disp(['Moving left servo to: ', num2str(leftServoPos)]);
disp(['Moving right servo to: ', num2str(rightServoPos)]);
end

function moveAndWait(maestro, motorID, servoPos, velocity, trialConfig)
maestro.setMotor(motorID, trialConfig.ConvertMaestroPos(servoPos), velocity);
while ~maestro.checkPosition(motorID, servoPos, 3)
end
end

function waitForBothSpouts(maestro, leftServoPos, rightServoPos)
leftDone = false;
rightDone = false;
while ~(leftDone && rightDone)
    rightDone = maestro.checkPosition(0, rightServoPos, 3);
    leftDone = maestro.checkPosition(1, leftServoPos, 3);
end
end