# Main Workflow

1. Run `opto_test_doric_202607` from the Bpod protocol folder.
2. Edit the Bpod parameter GUI.
3. Press Enter in MATLAB to validate settings and configure the display.
4. Confirm the Doric opto settings printed in MATLAB.
5. The protocol opens the PsychToolbox window and sets it to gray.
6. Press Enter again to start trials.
7. Each trial syncs GUI parameters, assigns opto for the next trial, builds a state machine, runs Bpod, saves data, and updates the session canvas.
8. When the session ends, the display is returned to gray and closed.

## What Can Change During A Session

The GUI is synced before each trial. Changes affect the next trial.

Completed trial data are not rewritten. Completed opto assignments remain in `BpodSystem.Data.OptoTrialTypes`.

## Visual Stimulus

The protocol loads root `image.png` if it exists. If it does not exist, `GenerateVisualCueVideo.m` creates a sinusoidal grating. The same frame is loaded into video slots 1 and 2, so image 1 and image 2 match.

`ImageDuration_s` is rounded to the nearest whole number of display frames after the PsychToolbox display reports its frame rate.
