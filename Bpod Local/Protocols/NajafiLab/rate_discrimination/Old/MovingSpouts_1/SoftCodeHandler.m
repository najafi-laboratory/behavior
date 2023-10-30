function SoftCodeHandler(code)
global BpodSystem
global S;
global M

%S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S
%S = struct;
%S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin

switch true
    case code == 255
        BpodSystem.PluginObjects.V.stop;
    case code >= 0 && code <=6
        %BpodSystem.PluginObjects.V.tic1 = tic; %% need to factor in processing overhead from whence play is called
        BpodSystem.PluginObjects.V.play(code);
    case code == 8
        % M.setMotor(0, -0.225);
        % M.setMotor(1, 0);
        
        % RightSpoutStart = 1350;
        % LeftSpoutStart = 
        

        M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos + S.GUI.ServoDeflection));
        M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos - S.GUI.ServoDeflection));
    case code == 9
        % M.setMotor(0, -0.455);
        % M.setMotor(1, 0.23);

        M.setMotor(0, ConvertMaestroPos(S.GUI.RightServoInPos));
        M.setMotor(1, ConvertMaestroPos(S.GUI.LeftServoInPos)); 


end
% 
% if code ~= 255
%     %BpodSystem.PluginObjects.V.tic1 = tic; %% need to factor in processing overhead from whence play is called
%     BpodSystem.PluginObjects.V.play(code);
% elseif code
%     BpodSystem.PluginObjects.V.stop;
% endBpodBB

function SetMotorPos = ConvertMaestroPos(MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;