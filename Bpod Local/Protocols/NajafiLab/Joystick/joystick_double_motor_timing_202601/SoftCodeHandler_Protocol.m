function SoftCodeHandler_Protocol(code)
global BpodSystem
global S
global M
global ProtocolTrialContext

% Dispatch hardware actions requested by state-machine soft codes.
switch code
    case 0
        stopCue;
    case {1, 2}
        playCue(code);
    case 3
        showGray;
    case 7
        M.setMotor(0, maestroPosition(S.GUI.ServoInPos - S.GUI.ServoOutPos));
        SendBpodSoftCode(1);
    case 8
        moveServoHome;
        SendBpodSoftCode(2);
    case 9
        moveServoHome;
    case 12
        stopCue;
        M.setMotor(0, maestroPosition(S.GUI.ServoInPos - S.GUI.ServoOutPos));
        SendBpodSoftCode(1);
    case 18
        stopCue;
        moveServoHome;
        SendBpodSoftCode(2);
    case 19
        ProtocolTrialContext.Press2Clock = tic;
    case 20
        deliverDynamicReward;
        SendBpodSoftCode(3);
    case 21
        capturePress2;
        stopCue;
    case 22
        moveServoHome;
        SendBpodSoftCode(press2OutcomeCode());
end

    function playCue(index)
        % Present one visual cue while keeping the sync patch light.
        stopVideo;
        BpodSystem.PluginObjects.V.play(index);
    end

    function stopCue
        % Return display to gray and make the sync patch dark.
        stopVideo;
        BpodSystem.PluginObjects.V.setSyncPatch(0);
    end

    function showGray
        stopVideo;
        BpodSystem.PluginObjects.V.setSyncPatch(0);
    end

    function stopVideo
        try
            BpodSystem.PluginObjects.V.stop;
        catch exception
            if ~contains(exception.message, 'not running')
                rethrow(exception)
            end
        end
    end

    function moveServoHome
        % Retract servo and wait for the joystick to settle near zero.
        M.setMotor(0, maestroPosition(S.GUI.ServoInPos), 0.5);
        startTime = tic;
        while abs(BpodSystem.PluginObjects.R.currentPosition) > S.GUI.RetractThreshold && toc(startTime) < S.GUI.ServoReturnTimeout_s
            pause(0.001);
        end
    end

    function capturePress2
        % Convert press 2 time into reward amount using the GUI reward shape.
        if isempty(ProtocolTrialContext.Press2Clock)
            ProtocolTrialContext.Press2Time_s = NaN;
            ProtocolTrialContext.RewardAmount_uL = 0;
            return
        end
        pressTime = toc(ProtocolTrialContext.Press2Clock);
        difference = pressTime - ProtocolTrialContext.Delay;
        if difference < 0
            fraction = 1 + difference / ProtocolTrialContext.RewardWindowLeft_s;
        elseif difference <= ProtocolTrialContext.RewardMaximumWindow_s
            fraction = 1;
        else
            fraction = 1 - (difference - ProtocolTrialContext.RewardMaximumWindow_s) / ProtocolTrialContext.RewardWindowRight_s;
        end
        ProtocolTrialContext.Press2Time_s = pressTime;
        ProtocolTrialContext.RewardAmount_uL = ProtocolTrialContext.MaximumReward_uL * max(0, min(1, fraction));
    end

    function code = press2OutcomeCode
        % Report early, rewarded, or late to the state machine.
        difference = ProtocolTrialContext.Press2Time_s - ProtocolTrialContext.Delay;
        if difference < -ProtocolTrialContext.RewardWindowLeft_s
            code = 1;
        elseif difference > ProtocolTrialContext.RewardMaximumWindow_s + ProtocolTrialContext.RewardWindowRight_s
            code = 3;
        else
            code = 2;
        end
    end

    function deliverDynamicReward
        % Deliver the dynamically computed reward through valve 2.
        amount = ProtocolTrialContext.RewardAmount_uL;
        if amount <= 0
            return
        end
        valveTime = GetValveTimes(amount, 2);
        ManualOverride('OV', 2);
        valveCleanup = onCleanup(@() ManualOverride('OV', 2));
        pause(valveTime);
        clear valveCleanup
    end

    function position = maestroPosition(value)
        position = value * 0.002 - 3;
    end
end
