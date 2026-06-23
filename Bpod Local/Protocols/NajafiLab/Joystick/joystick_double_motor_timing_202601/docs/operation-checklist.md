# Operation Checklist

Use this checklist before and during a session.

## Before Starting

1. Open Bpod.
2. Confirm the correct subject is selected.
3. Confirm the rig computer is supported.
4. Confirm the Maestro is connected.
5. Confirm the rotary encoder is connected.
6. Confirm the display is connected.
7. Confirm the water line is ready.
8. Confirm Doric opto hardware is ready if opto is enabled.

## Start The Protocol

1. Run `joystick_double_motor_timing_202601`.
2. Edit GUI parameters.
3. Move the GUI if needed.
4. Press Enter in MATLAB.
5. Read the Doric opto settings printed in MATLAB.
6. Match Doric frequency and pulse on time.
7. Press Enter to confirm.
8. Wait for the gray screen.
9. Press Enter again to start trials.

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

### No visual cue appears

Check monitor selection.

Check PsychToolbox display.

Check `UseGeneratedGrating`.

Check `image.png` if using the image cue.

### Opto timing looks wrong

Check `OptoMode`.

Check the enabled opto type fields.

Check Doric frequency and pulse on time.

Check LED1 row in the event plot.

### Reward amount is zero

The trial may be a reward omission probe.

The press may be outside the reward window.

The reward window settings may be too narrow.
