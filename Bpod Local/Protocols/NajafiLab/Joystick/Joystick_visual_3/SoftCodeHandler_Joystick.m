function SoftCodeHandler_Joystick(code)
global BpodSystem
global S

switch true
    case code == 7
        BpodSystem.PluginObjects.S.holdRMScurrent = 200; % Immobilize the lever
        
        Tolerance = 1; % Lever is home if within this Tolerance of 0, unit = degrees
        currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder        
        ramp = 0; % stepping increment to ramp the reset speed
        disp(['S.GUI.ZeroRTrials = ' num2str(S.GUI.ZeroRTrials)]);
        disp(['S.GUI.currentTrial = ' num2str(S.GUI.currentTrial)]);
        while abs(currentPosition) > Tolerance
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
        
        if S.GUI.currentTrial > S.GUI.ZeroRTrials
            BpodSystem.PluginObjects.S.holdRMScurrent = S.GUI.ResistanceLevel; % set res
            BpodSystem.PluginObjects.S
            disp(['S.GUI.ResistanceLevel = ' num2str(S.GUI.ResistanceLevel)]);
        else
            
            BpodSystem.PluginObjects.S.holdRMScurrent = 0; % set res
            BpodSystem.PluginObjects.S
            disp('0');
        end
        SendBpodSoftCode(1); % Indicate to the state machine that the lever is back in the home position        
    case code == 8
        disp('code 8');        
        BpodSystem.PluginObjects.S.holdRMScurrent = 200; % Immobilize the lever        
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
        BpodSystem.PluginObjects.S.holdRMScurrent = 200; % Immobilize the lever
        
        Tolerance = 1; % Lever is home if within this Tolerance of 0, unit = degrees
        currentPosition = BpodSystem.PluginObjects.R.currentPosition; % Read the encoder
        futurePosition = currentPosition + S.GUI.Threshold + 1;
        ramp = 0; % stepping increment to ramp the reset speed
        disp(['S.GUI.ZeroRTrials = ' num2str(S.GUI.ZeroRTrials)]);
        disp(['S.GUI.currentTrial = ' num2str(S.GUI.currentTrial)]);
        disp(['abs(futurePosition - currentPosition) = ' num2str(abs(futurePosition - currentPosition))]);
        while abs(futurePosition - currentPosition) > Tolerance
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




        if S.GUI.currentTrial > S.GUI.ZeroRTrials
            BpodSystem.PluginObjects.S.holdRMScurrent = S.GUI.ResistanceLevel; % set res
            BpodSystem.PluginObjects.S
            disp(['S.GUI.ResistanceLevel = ' num2str(S.GUI.ResistanceLevel)]);
        else
            
            BpodSystem.PluginObjects.S.holdRMScurrent = 0; % set res
            BpodSystem.PluginObjects.S
            disp('0');
        end
        SendBpodSoftCode(1); % Indicate to the state machine that the lever is back in the home position          
    case code >= 0 && code <= 6 
        BpodSystem.PluginObjects.V.play(code);
    case code == 255
        BpodSystem.PluginObjects.V.stop;

end


function steps = degrees2MotorSteps(degrees, nMotorStepsPerRev)
steps = round((degrees/360)*nMotorStepsPerRev);