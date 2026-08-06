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
        showIdleScreen;
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
        % Flip the sync patch light, then trigger the preloaded audio.
        showIdleScreen;
        BpodSystem.PluginObjects.V.play(index);
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
        % Audio-only uses its dedicated black/dark-patch frame, never gray.
        stopAudio;
        if audioOnlyIdleAvailable()
            BpodSystem.PluginObjects.V.play(audioOnlyIdleSlot());
        else
            stopVideo;
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

    function yes = audioOnlyIdleAvailable
        yes = isAudioOnly() && numel(BpodSystem.PluginObjects.V.Videos) >= audioOnlyIdleSlot() && ...
            ~isempty(BpodSystem.PluginObjects.V.Videos{audioOnlyIdleSlot()});
    end

    function yes = isAudioOnly
        yes = isfield(S.GUI, 'SensoryCueMode') && S.GUI.SensoryCueMode == 2;
    end

    function yes = hifiAvailable
        yes = isfield(BpodSystem.PluginObjects, 'H') && ~isempty(BpodSystem.PluginObjects.H);
    end

    function yes = soundCardAvailable
        yes = isfield(BpodSystem.PluginObjects, 'Sound') && ~isempty(BpodSystem.PluginObjects.Sound);
    end

    function slot = audioOnlyIdleSlot
        slot = 3;
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
        % Spread calibrated valve-on time evenly across the reward window.
        amount = ProtocolTrialContext.RewardAmount_uL;
        if amount <= 0
            return
        end
        valveTime = GetValveTimes(amount, 2);
        totalDuration = ProtocolTrialContext.TotalRewardDuration_s;
        if totalDuration <= valveTime
            deliverValvePulse(valveTime);
            return
        end

        % Use approximately one calibrated valve-time per cycle. Limit the
        % count so no valve-on command is shorter than MATLAB's useful 1 ms
        % scheduling resolution. Every cycle has the same duty cycle.
        minimumPulse_s = 0.001;
        cycleCount = max(1, round(totalDuration / valveTime));
        cycleCount = min(cycleCount, max(1, floor(valveTime / minimumPulse_s)));
        cycleDuration = totalDuration / cycleCount;
        pulseDuration = valveTime / cycleCount;
        offDuration = cycleDuration - pulseDuration;

        rewardClock = tic;
        for cycle = 1:cycleCount
            deliverValvePulse(pulseDuration);
            % Pause to the absolute cycle boundary so valve-command overhead
            % does not accumulate and stretch the configured total window.
            remainingCycleTime = cycle * cycleDuration - toc(rewardClock);
            if offDuration > 0 && remainingCycleTime > 0
                pause(remainingCycleTime);
            end
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
