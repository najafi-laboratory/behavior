# Operation Checklist

Use this checklist before and during a session.

## Before Starting

1. Open Bpod.
2. Confirm the correct subject is selected.
3. Confirm the rig computer is supported.
4. Confirm the Maestro is connected.
5. Confirm the rotary encoder is connected.
6. Confirm the display is connected.
7. Confirm the HiFi module or current system speaker is available if using audio cues.
8. Confirm the water line is ready.
9. Confirm Doric opto hardware is ready if opto is enabled.

## Start The Protocol

1. Run `joystick_double_motor_timing_202601`.
2. Confirm the Bpod console status LED turns off.
3. Edit GUI parameters.
4. Move the GUI if needed.
5. Press Enter in MATLAB.
6. Read the Doric opto settings printed in MATLAB.
7. Match Doric frequency and pulse on time.
8. Press Enter to confirm.
9. At the ready prompt, press Enter again to start trials.

## During The Session

Watch these plots:

- Trial type raster.
- Opto raster.
- Probe raster.
- Press 2 timing histograms.
- Outcome fractions.
- Encoder trace.
- State timing.
- Event plot.

Check these signs:

- BNC1 goes high during trials.
- LED1 appears on opto trials.
- Encoder trace crosses threshold on presses.
- Reward amount is printed after reward trials.
- No-press outcomes appear when expected.

## After The Session

The protocol should:

- Move the servo home.
- Close the visual display.
- Save session data.
- Print a session summary.

## Common Problems

### MATLAB stops with a settings error

Read the error text.

Fix the GUI parameter.

Restart the protocol.

### Rotary encoder is not found

Check USB connection.

Close other MATLAB serial objects.

Restart Bpod or MATLAB if needed.

### Servo does not move

Check Maestro USB.

Check rig mapping.

Check `ServoInPos` and `ServoOutPos`.

### No sensory cue appears

Check monitor selection.

Check PsychToolbox display.

Check `SensoryCueMode`.

For visual cues, check `UseGeneratedGrating`.

For image-based visual cues, check `image.png`.

For audio cues, check the HiFi module or current system speaker and `AudioStimVolume`. `AudioAttenuation_dB` applies only to HiFi.

### Opto timing looks wrong

Check `OptoMode`.

Check the enabled opto period checkboxes.

Check Doric frequency and pulse on time.

Check LED1 row in the event plot.

### Reward amount is zero

The trial may be a reward omission probe.

The press may be outside the reward window.

The reward window settings may be too narrow.
