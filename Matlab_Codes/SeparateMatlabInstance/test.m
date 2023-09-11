% Screen('Preference', 'SkipSyncTests', 1);
% Screen('OpenWindow', 2, 0);

Port = 1;
T = TCPCom(Port); 

while 1
    try
        msg = T.read(1);  % read 1 byte
        disp(msg);
    catch err
        disp(err);
    end

    % Abort demo if any key is pressed:
    if KbCheck
        break;
    end
end