function fnActivateAirPuffPulse(obj, source,event)
    global BpodSystem
    global S
    
    % Get current air puff pulse duration
    S = BpodParameterGUI('sync', S);

    % Create air puff pulse timer if needed
    if isempty(BpodSystem.Data.AirPuffPulseTimer) || ~isvalid(BpodSystem.Data.AirPuffPulseTimer)
        BpodSystem.Data.AirPuffPulseTimer = timer('TimerFcn',@(x,y)ToggleOutputState(), 'ExecutionMode', 'singleShot', 'StartDelay', S.GUI.AirPuff_Pulse_Dur);
    end

    % Process air puff pulse toggle if the timer isn't already running
    if ~strcmp(BpodSystem.Data.AirPuffPulseTimer.Running, 'on')
        % Update start delay using current air puff pulse duration
        BpodSystem.Data.AirPuffPulseTimer.StartDelay = S.GUI.AirPuff_Pulse_Dur;             

        % start air puff
        ToggleOutputState();

        % set timer to stop air puff
        start(BpodSystem.Data.AirPuffPulseTimer);   
    end
end

% toggle output port state, currently only for Valve 1
function ToggleOutputState()
    global BpodSystem
    global S

    Ch = 14; % Valve 1 - Air Puff
    BpodSystem.HardwareState.OutputState(Ch) = 1-BpodSystem.HardwareState.OutputState(Ch);
    DigitalOutputChannel = Ch-1;
    OverrideMessage = ['O' DigitalOutputChannel BpodSystem.HardwareState.OutputState(Ch)];
    BpodSystem.SerialPort.write(OverrideMessage, 'uint8');
end


