M = PololuMaestro('COM15');

M.setMotor(0, 0);
pause(0.3);
M.setMotor(0, 0.5);
pause(0.3);
M.setMotor(0, 0);


pause(0.1);
M = [];