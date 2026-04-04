"""
constants.py

Contains all global hyperparameters, simulation constants, and tuning
weights used across the behavior modeling pipeline. Centralizing these
prevents hard-coding and makes hyperparameter sweeps easier.
"""

# --- Optimization Constants ---
N_RESTARTS = 40                     # Number of random initializations for fitting
OPTIM_STEPS_STAGE_1 = 3000         # Steps for sensory parameter lead stage
OPTIM_STEPS_STAGE_2 = 3000         # Steps for learning parameter lead stage
OPTIM_STEPS_STAGE_3 = 3000         # Steps for joint fine-tuning stage
LEARNING_RATE = 0.0005             # Base learning rate for Adam optimizer

# --- Physical Model Constants ---
PHYSICAL_BOUNDARY_SECONDS = 1.25   # Initial physical time boundary for the interval

# --- Penalty & Loss Constants ---
RARE_TRIAL_WEIGHT    = 2.5         # Importance multiplier for rare/surprising trials
BARRIER_K            = 0.01        # Soft barrier strength (~1-2 % of typical NLL at init)
BARRIER_MARGIN       = 0.04        # Fraction of param range where barrier activates
TRANSITION_LOSS_WEIGHT = 8.0       # Weight for transition-asymmetry term in restart selection
N_POST_SWITCH        = 20          # Trials after a block switch to include in transition curve