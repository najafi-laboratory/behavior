# Workflow

## Launch

Run `block_single_interval_discrimination_202606` from the protocol folder. The protocol removes subfolders from the MATLAB path so the rewritten protocol does not import code from `ref1` or `ref2`.

## GUI Edit Pass

The protocol builds defaults with `ConfigureProtocol`, opens `BpodParameterGUI`, and waits for the first MATLAB Enter prompt. Edit parameters before pressing Enter.

The GUI sections are arranged into balanced columns by panel size:

1. ITI, Chemo
2. Reward, OptoSchedule, Choice
3. OptoPeriods, Audio
4. ISI, Stimulus
5. Blocks, Probe, Session
6. OptoHardware, Servo

Blocks, Audio, opto schedule, opto hardware, opto periods, and Chemo have separate panels so the longer session setup is easier to scan.

## Hardware Setup

After the first Enter:

1. Settings are synced and validated.
2. HiFi is opened and configured.
3. The PsychToolbox video player is opened.
4. A grey ready video is loaded.
5. Maestro servo is opened and spouts move out.

## Second Enter

Before trial start, the display uses the same grey handling pattern as the reference protocol:

1. Stop active video.
2. Set sync patch dark.
3. Repeatedly call `V.play(0)` while waiting for Return.

No cursor movement or keyboard capture is used.

## Trial Loop

For each trial:

1. Sync GUI parameters.
2. Sample ISI, ITI, and punish ITI.
3. Select target side from trial type and contingency.
4. Build and load stimulus videos and audio.
5. Build and send state machine.
6. Run state machine.
7. Save raw events and derived trial fields.
8. Update plots.
9. Return screen to grey.

If the trial is tagged for opto and the session is trained, the state machine arms the selected period timers. Stimulus opto spans from `PreStimDelay` onset through stimulus-play offset. Spout-in-delay opto spans `SpoutInDelay`. Spout-in opto spans `ChoiceWindow`, `ProbeChoiceWindow`, or naive `WaitForCorrectLick`. Pre-outcome opto spans `PreOutcomeDelay` or `PreOutcomeDelayPunish`. Reward opto spans `Reward`. Post-reward opto spans `PostRewardDelay`. Punish-ITI opto spans `PunishITI`.

Opto settings are synced at the beginning of each trial. The opto plot shows the initial intended schedule as small dots and the online assigned trial settings as solid squares.

## Cleanup

When the protocol ends, cleanup stops HiFi/video objects, moves the spouts out when possible, closes session figures, and returns control to Bpod.
