# Operation Checklist

## Before Starting

1. Confirm Bpod `PWM1` / LED1 is connected to the Doric external gate or trigger input.
2. Configure Doric to generate the desired waveform when the external input is high.
3. Decide whether Bpod should gate the full epoch or only a fixed duration from epoch onset.
4. Set `OptoFrequency_Hz` and `OptoPulseOn_ms` in the GUI to match Doric for operator record keeping.
5. Confirm `OptoFraction` and the enabled opto epochs.
6. Confirm pre-stim and post-stim delays are positive if their opto checkboxes are enabled.

## During The Session

1. Watch the opto trial type raster for assigned opto trials.
2. Watch completed state timing to confirm the trial sequence.
3. Watch event timing to confirm `BNC1` and LED1 timing.
4. GUI changes affect the next trial after the current trial finishes.

## After The Session

1. Confirm `BpodSystem.Data.OptoTrialTypes` has four rows.
2. Confirm `BpodSystem.Data.ITI` matches completed trials.
3. Save or export the Bpod session as usual.
