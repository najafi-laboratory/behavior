function SoftCodeHandler_BlockSingleInterval(code)
% Handle video soft codes for the AV interval protocol.
global BpodSystem
global S
global M

switch code
    case 8
        moveSpoutsOut;
        SendBpodSoftCode(2);
    case 9
        moveSpoutsIn;
        SendBpodSoftCode(2);
    case 25
        playVideo(25);
    case 26
        playVideo(26);
    case {254, 255}
        playGray;
end
end

function moveSpoutsIn
global S
global M

if isempty(M)
    return
end
M.setMotor(0, maestroPosition(S.GUI.RightServoInPos - S.GUI.ServoDeflection), S.GUI.ServoVelocity);
M.setMotor(1, maestroPosition(S.GUI.LeftServoInPos + S.GUI.ServoDeflection), S.GUI.ServoVelocity);
end

function moveSpoutsOut
global S
global M

if isempty(M)
    return
end
M.setMotor(0, maestroPosition(S.GUI.RightServoInPos), S.GUI.ServoVelocity);
M.setMotor(1, maestroPosition(S.GUI.LeftServoInPos), S.GUI.ServoVelocity);
end

function position = maestroPosition(value)
position = value * 0.002 - 3;
end

function playVideo(index)
global BpodSystem

stopVideo;
BpodSystem.PluginObjects.V.TimerMode = 1;
BpodSystem.PluginObjects.V.play(index);
end

function playGray
global BpodSystem

stopVideo;
try
    BpodSystem.PluginObjects.V.setSyncPatch(0);
catch
end
end

function stopVideo
global BpodSystem

try
    BpodSystem.PluginObjects.V.stop;
catch exception
    if ~contains(exception.message, 'not running')
        rethrow(exception)
    end
end
end
