% MATLAB: send 1-byte commands to Arduino
% Test serial comms with Arduino_Sig_1

port = "COM4";      % <-- change to your COM port
baud = 115200;

s = serialport(port, baud);

% IMPORTANT: Arduino often resets when you open the port
pause(2.0);

% Send PWM ON command (0x01)
write(s, uint8(1), "uint8");

pause(1.0);

% Send PWM OFF command (0x02)
write(s, uint8(2), "uint8");

pause(0.2);

% Optional: ping (0xA5) and read reply (0x5A)
% write(s, uint8(hex2dec('A5')), "uint8");
pause(0.05);

if s.NumBytesAvailable > 0
    reply = read(s, 1, "uint8");
    fprintf("Reply byte: 0x%02X\n", reply);
else
    disp("No reply received.");
end

% Clean up
clear s
