# Joystick Double Motor Timing Protocol

This is a compact Bpod MATLAB protocol for a joystick timing task with visual cues, servo control, rotary encoder thresholds, optional assist trials, opto/probe trial tags, and online plotting.

## Main Workflow

1. Run `joystick_double_motor_timing_202601`.
2. The GUI opens first. Set parameters and press Enter in MATLAB.
3. Hardware is configured: Pololu Maestro servo, rotary encoder, and PsychToolbox video display.
4. The screen is set to gray and the servo returns home. Press Enter again to start trials.
5. Each trial syncs GUI parameters, builds the next state machine, runs Bpod, saves trial data, and updates the plot canvas.

## Trial Logic

The state machine starts with `VisualCue1`, then either waits for `Press1` in double-press mode or goes directly to press 2 in single-press mode.

For press 2, timing is measured from the start of the press 2 window. The state `Press2` captures the timing and `ServoBack2` routes the trial to `Press2Early`, `RewardDelay`, or `Press2Late`. Missing presses go to `DidNotPress1` or `DidNotPress2`. Rewarded trials finish through `Reward`, `PostRewardDelay`, and `OutcomeReward`.

Assist trials hold the servo until perfect timing, release it, and then use the same `Press2` outcome path.

## Reward Shape

Reward amount is computed in `SoftCodeHandler_Protocol` from the press 2 time:

- zero at `perfect timing - RewardWindowLeft_s`
- maximum at perfect timing
- maximum through `RewardMaximumWindow_s`
- linearly decreases to zero over `RewardWindowRight_s`

## Main Files

- `joystick_double_motor_timing_202601.m`: main protocol, GUI sync, hardware setup, trial loop, data saving.
- `ConfigureProtocol.m`: GUI defaults, metadata, and parameter panels.
- `BuildStateMachine.m`: Bpod state machine for each trial.
- `SoftCodeHandler_Protocol.m`: servo, visual cue, press timing, and reward delivery soft-code operations.
- `GenerateTrials.m`: short/long trial blocks plus ITI and punish ITI generation.
- `OptoControl.m`: opto trial tags and display colors.
- `ProbeControl.m`: probe trial tags and display colors.
- `GenerateVisualCueVideo.m`: creates cue video frames from `image.png`.
- `ProtocolPlot.m`: one online canvas for trial outcomes, opto/probe tags, delay, press timing, encoder, states, and BNC/lick events.

## Plot Notes

The event plot uses one row per signal: `BNC 1`, `BNC 2`, and lick ports 1 to 3. Filled bars mark logical 1 or detected lick pulses; blank space marks 0.
