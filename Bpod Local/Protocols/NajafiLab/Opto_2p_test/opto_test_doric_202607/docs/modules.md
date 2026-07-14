# Modules And Files

## `opto_test_doric_202607.m`

Main protocol loop. It loads settings, opens the GUI, validates opto settings, configures the PsychToolbox display, runs trials, saves data, updates the canvas, and closes the display.

## `ConfigureProtocol.m`

Defines the small GUI parameter set: session, timing, ITI, and opto controls.

## `GenerateTrials.m`

Generates ITI values from the reference manual/exponential ITI settings.

## `GenerateVisualCueVideo.m`

Loads optional root `image.png` or generates a sinusoidal grating, resizes it to the display viewport, and snaps the duration to display frames.

## `OptoControl.m`

Assigns opto trials from `OptoFraction`, returns display labels, and returns state-machine output actions for selected opto epochs.

Saved opto data are `4 x nTrials`:

- row 1: pre-stim delay
- row 2: image interval
- row 3: image 2
- row 4: post-stim delay

## `BuildStateMachine.m`

Builds the six trial states plus `Start`. It handles display soft codes, `BNC1`, `PWM1`, and fixed-duration opto timers.

## `SoftCodeHandler_Protocol.m`

Handles display soft codes:

- `1`: play image 1
- `2`: play image 2
- `3`: show gray

## `ProtocolPlot.m`

Maintains one session canvas with opto trial types, completed state timing, and BNC/LED event timing.
