# Troubleshooting

## `image.png` Missing

If `UseSavedImage` is checked and `image.png` is not in the protocol folder, validation stops before the session starts. Put `image.png` next to `BuildStimulus.m`, or uncheck `UseSavedImage`.

## Screen Shows Startup Before Second Enter

The ready screen uses the same style as the reference protocol: stop active video, set sync patch dark, and call `V.play(0)` during the wait loop. The protocol does not move the cursor or capture keyboard input.

If the wrong screen appears, check that the stimulus display is on the expected monitor and that another application is not covering the PsychToolbox window.

## No Audio

Check:

- HiFi module is connected.
- `StimulusMode` is not visual-only.
- `AudioStimVolume` is above zero.
- `AudioAttenuation_dB` is not too low.

## Choice Window Starts Too Early

`SpoutIn` and `ProbeSpoutIn` transition only on `Tup`, so the choice window starts after `ServoMoveDelay_s`. If timing looks early, check `ServoMoveDelay_s` and servo velocity.

## Serial Port Errors

HiFi and Maestro serial ports must not be open in another MATLAB object. Restarting MATLAB or deleting stale serial objects may be required if Windows keeps the port busy.
