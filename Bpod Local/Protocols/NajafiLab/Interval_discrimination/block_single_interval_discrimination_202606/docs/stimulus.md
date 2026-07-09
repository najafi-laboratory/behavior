# Stimulus

Stimuli are built in `BuildStimulus.m`.

## Visual Source

When `UseSavedImage` is off, the visual stimulus is a generated sinusoidal grating.

When `UseSavedImage` is on, the protocol reads:

```text
image.png
```

from the protocol folder. The image is resized to the current video viewport by indexed sampling. Grayscale, RGB, floating-point, and RGBA PNG images are supported. Alpha is discarded.

If `UseSavedImage` is enabled and `image.png` is missing, validation stops the protocol before trials begin.

## Visual Sequence

The visual stimulus sequence is:

```text
stimulus image -> grey ISI -> stimulus image
```

The duration of each stimulus image is `GratingDuration_s`. The grey interval duration is the trial's sampled ISI. Durations are snapped to whole video frames.

## Audio Sequence

The audio tone uses:

- `AudioStimFreq_Hz`
- `AudioStimVolume`
- `AudioSamplingRate_Hz`
- `AudioRamp_ms`

Audio always follows the full stimulus timing:

```text
tone -> silence matching ISI -> tone
```

This matches the visual stimulus timing and keeps audio-only, visual-only, and audio+visual trials the same duration.

## Stimulus Mode

- `Visual only`: visual sequence is shown; audio is silent.
- `Audio only`: sync-patch video is played while audio is delivered.
- `Audio + visual`: visual sequence and audio are both delivered.

The audio-only mode still uses the video pathway so the sync patch timing remains available.
