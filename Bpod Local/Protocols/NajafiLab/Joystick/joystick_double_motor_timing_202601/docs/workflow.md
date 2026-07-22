# Main Workflow

This page describes the full protocol flow.

## 1. Start The Protocol

Run:

```matlab
joystick_double_motor_timing_202601
```

The protocol creates a cleanup object first. This makes sure hardware is reset if the protocol exits.

## 2. Load Settings

The protocol calls `ConfigureProtocol`.

This function:

- Loads saved Bpod protocol settings.
- Creates missing settings.
- Migrates old setting names.
- Removes unused setting names.
- Builds GUI metadata.
- Builds GUI panels.

## 3. Show Parameter GUI

The protocol calls:

```matlab
BpodParameterGUI('init', S)
```

Then it moves the GUI to the top-left of the screen.

The user edits parameters.

The user presses Enter in MATLAB.

The protocol syncs the GUI:

```matlab
S = BpodParameterGUI('sync', S)
```

## 4. Validate Settings

The protocol calls `validateSettings`.

This checks:

- Trial count is valid.
- Delay values are positive.
- Press windows are positive.
- Reward windows fit inside press 2 windows.
- Servo timeout is positive.
- Opto and probe fractions are between 0 and 1.
- Block edge settings are nonnegative integers.
- Reward amounts are positive.
- ITI settings are valid.
- Opto period settings are valid.
- Assist mode is disabled when opto mode is enabled.

If a setting is unsafe, the protocol stops with an error.

## 5. Confirm Doric Opto Settings

The protocol prints opto settings.

The user confirms that the Doric square-wave generator matches the GUI.

The protocol does not generate pulse trains itself. It gates LED1 / PWM1. The Doric device controls pulse frequency and pulse width.

## 6. Detect Rig

The protocol reads the computer hostname.

It maps the hostname to a rig name.

This decides display monitor selection and hardware assumptions.

## 7. Open Hardware

The protocol opens:

- Pololu Maestro servo controller.
- Rotary encoder module.
- Optional HiFi module.
- PsychToolbox video player.

The protocol releases stale serial objects before opening new ones.

The encoder starts USB streaming.

The protocol sets the soft-code handler:

```matlab
BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_Protocol'
```

## 8. Build Sensory Cue

The protocol calls `GenerateSensoryCueVideo` for the visual cue component and loads a ramped sine tone into the HiFi module when it is available.

The visual component either:

- Loads and resizes `image.png`.
- Or creates a generated grating.

Audio-only cues use a neutral gray frame.

If the HiFi module is missing, the protocol prints a warning and continues without auditory cue output.

The cue duration is snapped to a whole number of display frames.

The actual duration replaces `SensoryCueDuration_s`.

## 9. Prepare Hardware

The protocol moves the servo home.

It shows a gray screen.

It waits for the user to press Enter again.

This second Enter starts the session.

## 10. Initialize Online Plot

The protocol calls `ProtocolPlot('init', ...)`.

The plot window shows:

- Trial type raster, opto period raster, probe type raster, delay values, and encoder trace in the left column.
- Press 2 timing histograms.
- Completed outcome fractions, all outcome fractions, and outcome legend across the top of the right column.
- State timing.
- BNC, LED, and lick events.

Outcome plots share one outcome color set. Opto, probe, delay, and event plots use neutral gray or black marks. State timing and encoder annotations use colored markers, while the encoder position trace is black.

## 11. Trial Loop

The trial loop runs until `MaxTrials` is reached or the user stops Bpod.

Each trial does these steps:

1. Sync GUI values.
2. Validate settings.
3. Regenerate trial types if block settings changed.
4. Regenerate ITI values if ITI settings changed.
5. Regenerate probe tags if probe settings changed.
6. Regenerate opto tags if opto settings or probe exclusions changed.
7. Reload sensory cue media if cue duration, cue mode, cue source, or audio settings changed.
8. Choose short or long delay.
9. Choose ITI and punish ITI.
10. Apply probe settings.
11. Decide if this is an assist trial. Opto sessions require assist mode off.
12. Compute the maximum possible reward.
13. Save per-trial reward context.
14. Print trial settings.
15. Update online plots.
16. Configure encoder threshold.
17. Build the state machine.
18. Send the state machine.
19. Run the state machine.
20. Save raw events.
21. Save trial settings and trial tags.
22. Save press 2 time.
23. Save reward amount.
24. Read encoder data.
25. Normalize encoder data.
26. Save session data.
27. Update online plots.
28. Handle pause.

## 12. End Session

At the end, the protocol:

- Moves the servo home.
- Stops the sensory cue media and visual display.
- Prints a session summary.
- Runs cleanup.

Cleanup also runs if MATLAB exits the function unexpectedly.
