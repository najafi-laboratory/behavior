% MATLAB: send 1-byte commands to Arduino
% Test serial comms with Arduino_Sig_1

port = "COM5";      % <-- change to your COM port
baud = 115200;

s = serialport(port, baud);
% configureTerminator(s, "none");
flush(s);

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
% write code 3, get acknowledge response
write(s, uint8(3), "uint8");
pause(0.05);

if s.NumBytesAvailable > 0
    reply = read(s, 1, "uint8");
    fprintf("Reply byte: 0x%02X\n", reply);
else
    disp("No reply received.");
end

% --- Request pulse count ---
write(s, uint8(4), "uint8");   % 0x04

% Read 4 bytes (uint32)
raw = read(s, 4, "uint8");

% Convert to uint32
pulseCount = typecast(uint8(raw), 'uint32');

disp(pulseCount);

% Clean up
clear s
