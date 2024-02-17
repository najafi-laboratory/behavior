function SoftCodeHandler_Joystick(code)
global BpodSystem
global S
global M

global LastKnownEncPos


switch true    
    case code == 7
        % BpodSystem.PluginObjects.S.holdRMScurrent = 30;
        % BpodSystem.PluginObjects.S.holdRMScurrent = 900; % 
        %BpodSystem.PluginObjects.S.holdRMScurrent = 30; % 
        % 
        %BpodSystem.PluginObjects.S.holdRMScurrent = 900;
        Tolerance = 0.2; % Lever is home if within this Tolerance of 0, unit = degrees
        BpodSystem.PluginObjects.S.MaxSpeed = 1;
        %currentPosition = 999;
        currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
        % if (currentPosition == 999)
        %     %currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
        %     currentPosition = LastKnownEncPos;
        %     %BpodSystem.PluginObjects.S.microStep(-5);
        % end       
        ramp = 0; % stepping increment to ramp the reset speed
        disp(['S.GUI.ZeroRTrials = ' num2str(S.GUI.ZeroRTrials)]);
        disp(['S.GUI.currentTrial = ' num2str(S.GUI.currentTrial)]);
        % while abs(currentPosition) > Tolerance
        %     %BpodSystem.PluginObjects.S.microStep(-1*degrees2MotorSteps(currentPosition, 51200)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
        %     % BpodSystem.PluginObjects.S.microStep(-1 - ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
        %     % ramp = ramp + 1;
        %     currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
        %     disp(['pos = ' num2str(currentPosition)]);
        %     ramp = ramp + 1;     
        %     microSteps = 50;
        %     if currentPosition > 0
        %         BpodSystem.PluginObjects.S.microStep(-microSteps); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step                
        %     else
        %         BpodSystem.PluginObjects.S.microStep(microSteps); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
        %     end
        %     %pause(.01); 
        % end
        
        previousPos = currentPosition;
        ramp = 0;
        microSteps = 100;
        disp(['beh bove pos = ' num2str(currentPosition)]);
        setBreakingCurrentFlag = 0;
        while abs(currentPosition) > Tolerance
            
            
            % if ~setBreakingCurrentFlag
            %     if (abs(currentPosition) - Tolerance) < 0.5
            %         %BpodSystem.PluginObjects.S.holdRMScurrent = 30;
            %         %BpodSystem.PluginObjects.S.holdRMScurrent = 900;
            %         BpodSystem.PluginObjects.S.MaxSpeed = 1; % set max speed
            %         microSteps = 10;
            %         setBreakingCurrentFlag = 1;
            %     end
            % end
            %BpodSystem.PluginObjects.S.microStep(-1*degrees2MotorSteps(currentPosition, 51200)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
            % BpodSystem.PluginObjects.S.microStep(-1 - ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
            % ramp = ramp + 1;
            currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
            %disp(['pos = ' num2str(currentPosition)]);
            
            %microSteps = 1;
            % if currentPosition == previousPos
            %     %ramp = ramp + 1; % ramp
            % end
            if currentPosition > 0
                BpodSystem.PluginObjects.S.microStep(-(microSteps + ramp)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step                
            elseif currentPosition < 0
                BpodSystem.PluginObjects.S.microStep(microSteps + ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
            else
                break
            end
            % previousPos = currentPosition;
            %pause(.03);
            BpodSystem.PluginObjects.S.holdRMScurrent = 900;
        end
        %BpodSystem.PluginObjects.S.MaxSpeed = 1000; % set max speed
        disp(['beh low pos = ' num2str(currentPosition)]);
        disp(['ramp = ' num2str(ramp)]);
        currentPosition
        
        % use zero res for first trial
        % if S.GUI.currentTrial > S.GUI.ZeroRTrials
        %     BpodSystem.PluginObjects.S.holdRMScurrent = S.GUI.ResistanceLevel; % set res
        %     BpodSystem.PluginObjects.S
        %     disp(['S.GUI.ResistanceLevel = ' num2str(S.GUI.ResistanceLevel)]);
        % else
        % 
        %     BpodSystem.PluginObjects.S.holdRMScurrent = 30; % set res
        %     BpodSystem.PluginObjects.S
        %     disp('0');
        % end
        BpodSystem.Data.TrialData{1, S.GUI.currentTrial}.LeverResetPos = [BpodSystem.Data.TrialData{1, S.GUI.currentTrial}.LeverResetPos currentPosition]; % store position to measure start pos for mouse press
        %BpodSystem.PluginObjects.S.holdRMScurrent = 30; % set lever break
        BpodSystem.PluginObjects.S
        %BpodSystem.PluginObjects.S.holdRMScurrent = 900; %
        SendBpodSoftCode(1); % Indicate to the state machine that the lever is back in the home position
    case code == 8  % no holding current
        disp('code 8');        
        BpodSystem.PluginObjects.S.holdRMScurrent = 0;   
        %BpodSystem.PluginObjects.S
        % get/set for stepper module vars, default values shown
        %BpodSystem.PluginObjects.S.Acceleration()  % get accel
        %BpodSystem.PluginObjects.S.Acceleration(800)  % set accel
        %BpodSystem.PluginObjects.S.MaxSpeed()  % get MaxSpeed
        %BpodSystem.PluginObjects.S.MaxSpeed(200)  % set MaxSpeed
        %BpodSystem.PluginObjects.S.PeakCurrent()   % get PeakCurrent
        %BpodSystem.PluginObjects.S.PeakCurrent(561)   % set PeakCurrent
        %BpodSystem.PluginObjects.S.RMScurrent() % get RMScurrent
        %BpodSystem.PluginObjects.S.RMScurrent(397) % set RMScurrent
        %BpodSystem.PluginObjects.S.ChopperMode()   % get chopper mode
        %BpodSystem.PluginObjects.S.ChopperMode(1)   % set chopper mode
        %BpodSystem.PluginObjects.S.BpodSystem.PluginObjects.S.MicroPosition()   % get micro position
        %BpodSystem.PluginObjects.S.BpodSystem.PluginObjects.S.MicroPosition(-1982763) set micro positiom, not sure about 'defaut', should be current pos, reseting would be like 'centering'
        %BpodSystem.PluginObjects.S.Position()   % get position
        %BpodSystem.PluginObjects.S.Position(-7.7452e+03)   % set position
        %BpodSystem.PluginObjects.S.resetPosition() % reset position to zero
        %BpodSystem.PluginObjects.S.EncoderPosition()   % get encoder position
        %BpodSystem.PluginObjects.S.resetEncoderPosition()  % reset encoder position
    case code == 9
        disp('code 9');        
        BpodSystem.PluginObjects.S.holdRMScurrent = 0; 
        %BpodSystem.PluginObjects.S
    case code == 10
        disp('code 10 prepress');   
        
    
    case code == 11
        disp('code 10');
        %BpodSystem.PluginObjects.S.holdRMScurrent = 200; % Immobilize the lever
        
        Tolerance = 1; % Lever is home if within this Tolerance of 0, unit = degrees
        currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
        futurePosition = currentPosition + S.GUI.Threshold + 1;
        ramp = 0; % stepping increment to ramp the reset speed
        disp(['S.GUI.ZeroRTrials = ' num2str(S.GUI.ZeroRTrials)]);
        disp(['S.GUI.currentTrial = ' num2str(S.GUI.currentTrial)]);
        disp(['abs(futurePosition - currentPosition) = ' num2str(abs(futurePosition - currentPosition))]);

        LeverGrabbed = 0; 
        while abs(futurePosition - currentPosition) > Tolerance

            % if BpodSystem.Status.LastEvent > 0
            %     if BpodSystem.Status.LastEvent <= length(BpodSystem.StateMachineInfo.EventNames)                    
            %         LastEvent = BpodSystem.StateMachineInfo.EventNames{BpodSystem.Status.LastEvent};
            %     end
            % end            
            % disp(['LastEvent = ' num2str(LastEvent)]);
            % disp(['BpodSystem.HardwareState.InputState = ' num2str(BpodSystem.HardwareState.InputState)]);
             
            Port = 3; % chan 3 nuerolink PI dev saunter bpoil slipstacker spec lim brain(brain) as ts->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><><><><><><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>><><<><>>>><><><>><><><<><><<><><><>><>
            PartReed = ReadBpodInput('Port', Port); % para bncstate
            if PartReed
                LeverGrabbed = 1;
            end

            %BpodSystem.PluginObjects.S.microStep(-1*degrees2MotorSteps(currentPosition, 51200)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
            % BpodSystem.PluginObjects.S.microStep(-1 - ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
            % ramp = ramp + 1;
            currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
            %disp(['pos = ' num2str(currentPosition)]);
            ramp = ramp + 1;
            if futurePosition - currentPosition < 0
                BpodSystem.PluginObjects.S.microStep(-1 - ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step                
            else
                BpodSystem.PluginObjects.S.microStep(1 + ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
            end
            pause(.01);
        end
        

        while abs(currentPosition) > Tolerance
            Port = 3; % chan 3 nuus
            PartReed = ReadBpodInput('Port', Port); % para bncstate
            if PartReed
                LeverGrabbed = 1;
            end

            %BpodSystem.PluginObjects.S.microStep(-1*degrees2MotorSteps(currentPosition, 51200)); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
            % BpodSystem.PluginObjects.S.microStep(-1 - ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
            % ramp = ramp + 1;
            currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
            %disp(['pos = ' num2str(currentPosition)]);
            ramp = ramp + 1;
            if currentPosition > 0
                BpodSystem.PluginObjects.S.microStep(-1 - ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step                
            else
                BpodSystem.PluginObjects.S.microStep(1 + ramp); % Move back to pos 0. The 51200 value is 200 steps per rev * 256 microsteps per step
            end
            pause(.01);
        end

        disp(['LeverGrabbed = ' num2str(LeverGrabbed)]);

        % use zero res for first trial
        if S.GUI.currentTrial > S.GUI.ZeroRTrials
            BpodSystem.PluginObjects.S.holdRMScurrent = S.GUI.ResistanceLevel; % set res
            BpodSystem.PluginObjects.S
            disp(['S.GUI.ResistanceLevel = ' num2str(S.GUI.ResistanceLevel)]);
        else
            
            BpodSystem.PluginObjects.S.holdRMScurrent = 0; % set res
            BpodSystem.PluginObjects.S
            disp('0');
        end

        if LeverGrabbed
            SendBpodSoftCode(2); % Indicate to the state machine that the lever is back in the home position          
        else
            SendBpodSoftCode(1); % Indicate to the state machine that the lever is back in the home position          
        end
    case code == 12
        disp('code 12');        
        BpodSystem.PluginObjects.S
        M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - S.GUI.ServoOutPos));
        SendBpodSoftCode(1); % Indicate to the state machine that the lever is back in the home position
    case code == 13
        disp('code 13');
        BpodSystem.PluginObjects.S
        % M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos));
        M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos), 0.5);
        % M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos - 2)); % retract at max speed
        % M.setMotor(0, ConvertMaestroPos(S.GUI.ServoInPos), 0.15); % slow down prior to contacting lever
    case code >= 0 && code <= 6 
        BpodSystem.PluginObjects.V.play(code);
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