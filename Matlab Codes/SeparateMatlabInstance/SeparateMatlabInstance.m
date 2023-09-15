

%!matlab -batch ScriptToRun

%pause(5.0);

Port = 1;
T = [];
while isempty(T) && ~KbCheck
    pause(1.0);
    try
        T = TCPCom('localhost', Port);
    catch err
        disp("could not connect to surfer")
    end 
end


i = 1;
while ~KbCheck
    % Abort demo if any key is pressed:
    % if KbCheck
    %     break;
    % end

    T.write(1);
    disp(['i:', num2str(i)]);
    i = i + 1;
    

    pause(5.0);
end