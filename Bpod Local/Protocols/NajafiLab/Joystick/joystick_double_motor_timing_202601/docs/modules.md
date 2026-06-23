# Modules And Files

This page explains each active file.

## `joystick_double_motor_timing_202601.m`

This is the main protocol file.

It does the session workflow:

1. Load settings.
2. Open GUI.
3. Validate settings.
4. Confirm opto settings.
5. Open hardware.
6. Build visual cues.
7. Wait for the start key.
8. Generate trial schedules.
9. Run each Bpod trial.
10. Save data.
11. Update plots.
12. Clean up hardware.

It also contains helper functions for:

- Rig detection.
- Serial port cleanup.
- Rotary encoder connection.
- Maestro port detection.
- GUI positioning.
- Probe trial settings.
- Setting validation.
- Trial result printing.
- Session summary printing.
- Outcome classification.

## `ConfigureProtocol.m`

This builds the protocol settings.

It defines:

- GUI default values.
- GUI panel groups: `Session`, `Stimulus`, `Timing`, `Joystick`, `Reward`, `ITI`, and `Manipulation`.
- Popup menu labels.
- Checkbox fields.
- Config migration rules.

It also removes old settings that are no longer used.

## `GenerateTrials.m`

This makes three vectors:

- `trialTypes`
- `itiValues`
- `punishITIValues`

`trialTypes` contains:

- `1`: short trial.
- `2`: long trial.

ITI values can be manual or exponential.

Block modes alternate between short and long blocks.

## `GenerateVisualCueVideo.m`

This builds the cue frame.

It either:

- Loads `image.png`.
- Or creates a generated grating.

The frame is resized to the display viewport.

The cue duration is rounded to a whole number of display frames.

## `BuildStateMachine.m`

This builds the Bpod state matrix for one trial.

Inputs include:

- Current settings.
- Short or long delay.
- Trial type.
- Opto type.
- Assist status.
- ITI.
- Punish ITI.

The state matrix handles:

- Cue 1.
- Servo release.
- Press 1.
- Press 2.
- Reward delay.
- Reward.
- Post reward delay.
- Error states.
- ITI.

It also adds opto output actions for the current opto type.

## `SoftCodeHandler_Protocol.m`

This handles soft codes from the state machine.

It controls:

- Visual cues.
- Gray screen.
- Servo movement.
- Press 2 timing.
- Reward calculation.
- Valve opening.
- Outcome soft-code return.

Dynamic reward is computed here.

## `OptoControl.m`

This controls opto trial tags and opto actions.

It supports four opto types:

- `0`: Off.
- `1`: Visual cue 1.
- `2`: Delay after press 1.
- `3`: Post reward.

It also returns plot labels and neutral grayscale colors.

When `OptoMode` is off, every trial receives opto type `0`.

When `OptoMode` is on, `OptoControl('trials', S, trialTypes)` first finds eligible trials. It excludes the first block and excludes the first and last `OptoZeroEdgeTrials` trials of each later block. It then tags approximately `OptoFraction` of the eligible trials.

Only enabled opto epochs can be assigned:

- `EnableOptoVisualCue1` allows type `1`.
- `EnableOptoDelay` allows type `2`.
- `EnableOptoPostReward` allows type `3`.

If more than one epoch is enabled, tagged trials draw randomly from the enabled epoch types. The generated vector is saved as `BpodSystem.Data.PlannedOptoTrialTypes`, and each completed trial is saved in `BpodSystem.Data.OptoTrialTypes`.

`OptoControl('actions', S, optoType, delay, press2Window)` returns Bpod output actions for the current trial:

- Type `0` keeps LED1 off and cancels the opto timer at `Start`.
- Type `1` turns LED1 on in `VisualStimulus1` and turns it off when cue 1 exits through the next servo or press-2 state.
- Type `2` arms global timer 10 for the press-2 delay epoch. The timer starts from `LeverRetract1`, drives `PWM1` high, and is cancelled at `RewardLeverRetract` or when the trial leaves the press-2 path.
- Type `3` turns LED1 on in `PostRewardDelay` and turns it off in `LeverRetractFinal`.

The current implementation uses continuous LED1 high intervals for the selected epoch. `OptoFrequency_Hz` and `OptoPulseOn_ms` are GUI parameters reserved for pulsed stimulation, but the active state-machine action sends a sustained `PWM1` high value during the epoch.

The event plot reconstructs the LED1 row from `OptoTrialTypes` and state timing. This keeps plotting self-contained and does not require reading a separate reference implementation.

## `ProbeControl.m`

This controls probe trial tags and probe metadata.

It supports:

- `0`: Off.
- `1`: Reward omitted.
- `2`: Timing mode flipped.

It also returns plot labels and neutral grayscale colors.

## `ProtocolPlot.m`

This manages the online session figure.

It updates:

- Trial type raster.
- Opto type raster.
- Probe type raster.
- Short press 2 timing.
- Long press 2 timing.
- Completed outcome fractions.
- All outcome fractions.
- Outcome legend.
- Delay values.
- Encoder trace.
- State timing.
- BNC, LED, and lick events.

It also reconstructs LED1 intervals for plotting.

Outcome-related plots share the outcome palette. Opto, probe, delay, and event plots use neutral gray or black marks. State timing and encoder annotations use colored markers, and the rotary encoder position trace is black.
