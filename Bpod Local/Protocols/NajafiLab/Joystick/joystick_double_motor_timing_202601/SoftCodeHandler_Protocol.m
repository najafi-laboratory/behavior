function SoftCodeHandler_Protocol(code)
global BpodSystem
global S
global M
global ProtocolTrialContext
persistent displayedVideoState displayedVideoPlayer

% Dispatch hardware actions requested by state-machine soft codes.
switch code
    case 0
        stopCue;
    case {1, 2}
        playCue(code);
    case 3
        showIdleScreen;
    case 7
        M.setMotor(0, maestroPosition(S.GUI.ServoInPos - S.GUI.ServoOutPos));
        SendBpodSoftCode(1);
    case 8
        moveServoHome;
        SendBpodSoftCode(2);
    case 9
        resetVideoState;
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
        % Present the cue directly. An intermediate idle-screen flip is
        % unnecessary and can block in Psychtoolbox display synchronization.
        stopAudio;
        presentVideo(index);
        if cueUsesAudio()
            if hifiAvailable()
                BpodSystem.PluginObjects.H.play(5);
            else
                play(BpodSystem.PluginObjects.Sound);
            end
        end
    end

    function stopCue
        showIdleScreen;
    end


    function showIdleScreen
        % Use the preloaded gray/dark-patch frame between cues.
        stopAudio;
        if idleVideoAvailable()
            presentVideo(idleVideoSlot());
        else
            presentGrayScreen;
        end
    end

    function stopAudio
        if hifiAvailable()
            try
                BpodSystem.PluginObjects.H.stop;
            catch
            end
        end
        if soundCardAvailable()
            try
                stop(BpodSystem.PluginObjects.Sound);
            catch
            end
        end
    end

    function yes = cueUsesAudio
        yes = isfield(S.GUI, 'SensoryCueMode') && ismember(S.GUI.SensoryCueMode, [2 3]) && ...
            (hifiAvailable() || soundCardAvailable());
    end

    function yes = idleVideoAvailable
        yes = numel(BpodSystem.PluginObjects.V.Videos) >= idleVideoSlot() && ...
            ~isempty(BpodSystem.PluginObjects.V.Videos{idleVideoSlot()});
    end

    function yes = hifiAvailable
        yes = isfield(BpodSystem.PluginObjects, 'H') && ~isempty(BpodSystem.PluginObjects.H);
    end

    function yes = soundCardAvailable
        yes = isfield(BpodSystem.PluginObjects, 'Sound') && ~isempty(BpodSystem.PluginObjects.Sound);
    end

    function slot = idleVideoSlot
        slot = 3;
    end

    function presentGrayScreen
        synchronizeVideoState;
        if isequal(displayedVideoState, 0)
            return
        end
        player = BpodSystem.PluginObjects.V;
        Screen('FillRect', player.Window, 128);
        Screen('Flip', player.Window, 0, 0, 2);
        displayedVideoState = 0;
    end

    function presentVideo(index)
        % These slots contain one static texture. Draw it directly and use
        % dontsync=2 so a lost/stalled vertical blank cannot block Bpod's
        % state-machine event loop in SensoryCue1.
        synchronizeVideoState;
        if isequal(displayedVideoState, index)
            return
        end
        player = BpodSystem.PluginObjects.V;
        Screen('DrawTexture', player.Window, player.Videos{index}.Data(1));
        Screen('Flip', player.Window, 0, 0, 2);
        displayedVideoState = index;
    end

    function synchronizeVideoState
        player = BpodSystem.PluginObjects.V;
        if isempty(displayedVideoPlayer) || ~isequal(displayedVideoPlayer, player)
            displayedVideoPlayer = player;
            displayedVideoState = NaN;
        end
    end

    function resetVideoState
        displayedVideoState = NaN;
        displayedVideoPlayer = [];
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
        % Deliver one tenth of the requested water in each reward cycle.
        amount = ProtocolTrialContext.RewardAmount_uL;
        if amount <= 0
            return
        end
        totalDuration = ProtocolTrialContext.TotalRewardDuration_s;
        cycleCount = 10;
        cycleDuration = totalDuration / cycleCount;

        % Calibrate the water assigned to one cycle, convert its valve time
        % to a per-cycle duty cycle, and cap delivery at 100% duty cycle.
        amountPerCycle = amount / cycleCount;
        valveTimePerCycle = GetValveTimes(amountPerCycle, 2);
        dutyCycle = min(1, max(0, valveTimePerCycle / cycleDuration));
        pulseDuration = dutyCycle * cycleDuration;
        offDuration = cycleDuration - pulseDuration;

        for cycle = 1:cycleCount
            deliverValvePulse(pulseDuration);
            pause(offDuration);
        end
    end

    function deliverValvePulse(duration)
        % Toggle valve 2 and guarantee closure if pause is interrupted.
        ManualOverride('OV', 2);
        valveCleanup = onCleanup(@() ManualOverride('OV', 2));
        pause(duration);
        clear valveCleanup
    end

    function position = maestroPosition(value)
        position = value * 0.002 - 3;
    end
end
