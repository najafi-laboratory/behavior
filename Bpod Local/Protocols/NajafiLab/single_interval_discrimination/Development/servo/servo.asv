



S.GUI.RightServoInPos = 1138.00;
S.GUI.LeftServoInPos = 1902.00;

S.GUI.ServoDeflection = -100;
S.GUI.ServoVelocity = 1;

M = PololuMaestro('COM5');

% M.setMotor(0, ConvertMaestroPos(1525.25));
% 
% M.setMotor(0, ConvertMaestroPos(1217.00));

M.setMotor(0, ConvertMaestroPos(1280.50));

setPosition = 1280.50;
isAtSetPosition = false;

while ~isAtSetPosition
    position = M.getPosition(0);
    disp(position)
    isAtSetPosition = M.checkPosition(0,setPosition,3);
end

% M.setMotor(0, ConvertMaestroPos(1500));

position = M.getPosition(0);
