function SoftCodeHandler_Joystick(code)
global M
global S
global BpodSystem

switch true
    case code == 7
        % disp('code 7');        
        M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - S.GUI.ServoOutPos));        
        SendBpodSoftCode(1); % Indicate to the state machine that the horiz bar is open for press
    case code == 8
        % disp('code 8');
        M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos), 0.5);
        % pause(1); % 200ms after starting retract prior to ITI or next stim/press
        % Tolerance = 0.4; % Lever is home if within this Tolerance of 0, unit = degrees
        Tolerance = S.GUI.RetractThreshold;

        % might need to add something here to abort if current position
        % isn't retreived within 500 ms or so
        currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the enc

        while abs(currentPosition) > Tolerance
            currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder           
            % disp(['pos = ' num2str(currentPosition)]);
            
        end
        % pause(.01);
        SendBpodSoftCode(1); % Indicate to the state machine that the lever is back in the home position
    case code >= 0 && code <= 6 
        BpodSystem.PluginObjects.V.play(code);
    case code == 12
        % used to combined end of trial ITI with punish ITI
        BpodSystem.Data.EndOfTrialITI = BpodSystem.Data.EndOfTrialITI + S.GUI.PunishITI;
    case code == 13 
        % softcode used to retur lever AND get combined trial ITI
        % time to return lever is measured and subtracted from the combined
        % trial ITI, instead of state 'Tup', matlab pause is used so that
        % we can combine ITI and ITI_punish into a single state
        tic
        % disp('code 8');
        M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos), 0.5);
        % pause(1); % 200ms after starting retract prior to ITI or next stim/press
        % Tolerance = 0.4; % Lever is home if within this Tolerance of 0, unit = degrees
        Tolerance = S.GUI.RetractThreshold;

        % might need to add something here to abort if current position
        % isn't retreived within 500 ms or so
        currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the enc

        while abs(currentPosition) > Tolerance
            currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder           
            % disp(['pos = ' num2str(currentPosition)]);
            
        end   
        ITI_Offset = toc;
        pause(BpodSystem.Data.EndOfTrialITI - ITI_Offset);
        SendBpodSoftCode(3);
    % case code == 14
    %     tic;    % set tic to measure TimeToPress when entering Press1/Press2
    %     M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - S.GUI.ServoOutPos));        
    %     SendBpodSoftCode(4); % Indicate to the state machine that bar is open (specific to wait1/2)
    % case code == 15
    %     % TimeToPress = toc;
    %     % disp(['Press1']);
    %     % disp(['TimeToPress = ' num2str(TimeToPress)]);
    %     % disp(['TimeRemainingToPress = ' num2str(S.GUI.Press1Window_s - TimeToPress)]);
    % 
    %     % tic
    %     % pause(S.GUI.Press1Window_s - TimeToPress);
    %     % toc
    % 
    %     SendBpodSoftCode(5); % Indicate to the state machine that S.GUI.Press1Window_s - TimeToPress has elapsed (specific to press1)
    % case code == 16
    %     % TimeToPress = toc;
    %     % disp(['Press2']);
    %     % disp(['TimeToPress = ' num2str(TimeToPress)]);
    %     % disp(['TimeRemainingToPress = ' num2str(S.GUI.Press2Window_s - TimeToPress)]);
    % 
    %     % tic
    %     % pause(S.GUI.Press2Window_s - TimeToPress);
    %     % toc
    % 
    %     SendBpodSoftCode(6); % Indicate to the state machine that S.GUI.Press1Window_s - TimeToPress has elapsed (specific to press2)

    case code == 17
        % return lever 700ms after reward and EarlyPress
        % tic
        pause(0.7);
        % toc
        M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos), 0.5);
        % set servo to return, although not waiting in softcode to sense
        % zero position
        SendBpodSoftCode(7); % Indicate to the state machine that the lever is back in the home position
        

    case code == 255
        BpodSystem.PluginObjects.V.stop;
end
end

function steps = degrees2MotorSteps(degrees, nMotorStepsPerRev)
    steps = round((degrees/360)*nMotorStepsPerRev);
end

function SetMotorPos = ConvertMaestroPos(MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end