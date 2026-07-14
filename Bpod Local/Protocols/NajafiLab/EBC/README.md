# Eyeblink Conditioning (EBC) Protocol
**Najafi Lab**

This repository contains the MATLAB/Bpod protocol used for head-fixed mouse eyeblink conditioning (EBC) experiments, together with post-processing scripts and archived protocol versions.

---

# Repository Structure

```
EBC/
│
├── EBC_Opto_V_5_4/          # Current experimental protocol (ACTIVE)
│
├── EBC_PostProcess_V_5_3/   # Data analysis and visualization
│
├── Old/                     # Previous protocol versions
│
├── Old_PostProcessing/      # Previous analysis scripts
│
└── README.md
```

Only **EBC_Opto_V_5_4** should be used for new experiments unless instructed otherwise.

---

# Overview

The protocol performs classical delay eyeblink conditioning using:

- Bpod Gen2
- High-speed camera
- Rotary encoder
- Air puff module
- LED conditioned stimulus (CS)
- Optional optogenetic stimulation
- Optional probe trials
- Single-block or alternating short/long ISI paradigms

The protocol automatically

- initializes hardware,
- randomizes trials,
- records behavioral events,
- synchronizes camera/video,
- saves all trial parameters,
- logs every session.

---

# Experimental Timeline

Typical delay conditioning trial

```
ITI
 │
 ▼
LED (CS)
 │───────────────
                │
                ▼
            Air Puff (US)
                │────
```

The mouse learns to close its eye during the delay between the LED and air puff.

---

# Hardware Requirements

The protocol expects the following devices:

- Bpod Gen2
- Rotary Encoder Module
- Arduino Trigger Controller
- High-speed camera
- Air puff controller
- LED stimulus
- (Optional) Optogenetic laser

Verify that all devices are connected before starting MATLAB.

---

# Starting an Experiment

1. Open MATLAB.

2. Navigate to

```
behavior/
```

3. Launch Bpod.

4. Select

```
NajafiLab
    └── EBC
            └── EBC_Opto_V_5_4
```

5. Press **Run**.

6. Adjust GUI parameters.

7. Press **Enter** to begin acquisition.

---

# Main GUI Parameters

## Timing

| Parameter | Description |
|------------|-------------|
| ITI_Pre | Minimum inter-trial interval |
| ITI_Post | Post-trial interval |
| ITI_Extra | Additional randomized ITI |
| LED_Dur | LED duration |
| AirPuff_Dur | Air puff duration |

---

## Air Puff Delay

Depending on the experiment mode:

### Single Block

```
AirPuff_OnsetDelay_SingleBlock
```

Example

```
400 ms
```

---

### Alternating Short/Long

```
AirPuff_OnsetDelay_Short
```

Example

```
200 ms
```

```
AirPuff_OnsetDelay_Long
```

Example

```
400 ms
```

---

## Trial Sequence

The protocol supports multiple training paradigms.

| Option | Description |
|---------|-------------|
| singleBlock | All trials use one ISI |
| singleTransition_short_to_long | Switch once from short to long |
| singleTransition_long_to_short | Switch once from long to short |
| doubleBlock_shortFirst | Alternate short then long blocks |
| doubleBlock_longFirst | Alternate long then short blocks |
| doubleBlock_RandomFirst | Random first block |

---

## Block Parameters

| Parameter | Description |
|------------|-------------|
| BlockLength | Average trials per block |
| Margine | Random variation in block length |

Example

```
BlockLength = 50

Margin = ±5

Actual block length = 45–55 trials
```

---

# Warm-up Trials

The protocol begins with warm-up trials.

```
num_warmup_trials
```

These trials allow the animal to stabilize before normal training begins.

---

# Probe Trials

Probe trials can be enabled independently of training stage.

Parameters include

```
UseProbeTrials

probetrials_percentage_perBlock

num_initial_nonprobe_trials_perBlock

ProbeMinSeparation
```

The protocol guarantees

- initial lead-in trials without probes
- minimum spacing between probes
- randomized probe placement

---

# Optogenetics

Optional optogenetic stimulation can be enabled.

Parameters include

- OptoEnabled
- OptoSessionType
- OptoFraction
- OptoDuration
- OptoOnset
- OptoInitialTrials
- OptoMinSeparation

If disabled, all trials behave as standard EBC trials.

---

# Session Output

Each session automatically creates

- MAT file
- Trial event log
- GUI parameters
- Random session ID
- Video synchronization information

Every session receives a unique session ID to synchronize with video recordings.

---

# Data Analysis

Post-processing scripts are located in

```
EBC_PostProcess_V_5_3/
```

Typical analyses include

- Eyelid segmentation
- Fraction eyelid closure (FEC)
- CR detection
- Learning curves
- Trial-by-trial visualization
- Session averages
- Short vs long comparison
- Probe trial analysis

---

# Typical Workflow

```
Prepare Mouse
        │
        ▼
Connect Hardware
        │
        ▼
Launch MATLAB
        │
        ▼
Start Bpod
        │
        ▼
Configure GUI
        │
        ▼
Run Session
        │
        ▼
Save Data
        │
        ▼
Run Post-Processing
        │
        ▼
Generate Figures
```

---

# Adding New Protocol Versions

Create a new folder

```
EBC_Opto_V_X_Y
```

Do **not** overwrite previous versions.

Document all major changes.

Example

```
V5.2
    Added probe trials

V5.3
    Improved post-processing

V5.4
    Added warm-up trials
    Updated GUI
    Improved block transitions
```

---

# Troubleshooting

## Missing GUI parameter

Usually caused by loading an old settings file.

Delete

```
ProtocolSettings.mat
```

or initialize missing fields inside `InitGUI.m`.

---

## Hardware not detected

Verify

- Bpod connected
- Encoder connected
- Arduino connected
- Camera connected

Restart Bpod if necessary.

---

## Session does not start

Check

- all GUI parameters
- COM ports
- hardware initialization
- MATLAB path

---

# Notes for New Students

1. Never modify the active protocol directly.

2. Create a new version before making changes.

3. Test every modification using a dummy session.

4. Keep archived versions in the **Old/** folder.

5. Always commit major updates to GitHub.

---

# Contact

For protocol questions, contact the current maintainer of the EBC protocol in the Najafi Lab.
