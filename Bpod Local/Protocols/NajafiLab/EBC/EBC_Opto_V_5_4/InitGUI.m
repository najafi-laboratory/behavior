classdef InitGUI
    methods
        function [S] = SetParams(obj, BpodSystem)
            S = BpodSystem.ProtocolSettings;
            if isempty(fieldnames(S))  % Default settings
                % ------------------------
                % Core EBC params
                % ------------------------
                S.GUI.ExperimenterInitials = 'Initials';
                S.GUI.WheelDirection = 1;
                S.GUIMeta.WheelDirection.Style = 'popupmenu';
                S.GUIMeta.WheelDirection.String = {'Forward','Reverse'};
                S.GUI.WheelRadius = 0.075;   % wheel radius in meters
                S.GUI.ITI_Pre  = 1;
                S.GUI.ITI_Post = 3;
                S.GUI.ITI_Extra = 3;
                S.GUI.LED_OnsetDelay = 0;
                S.GUI.LED_Dur = 0.05;
                S.GUI.BlockLength = 50;
                S.GUI.Margine = 5;
                S.GUI.TrialTypeSequence = 2;
                S.GUIMeta.TrialTypeSequence.Style = 'popupmenu';
                S.GUIMeta.TrialTypeSequence.String = ...
                    {'singleBlock','singleTransition_short_to_long','singleTransition_long_to_short', ...
                     'doubleBlock_shortFirst','doubleBlock_longFirst','doubleBlock_RandomFirst'};
                S.GUI.AirPuff_Dur = 0.02;
                S.GUI.AirPuff_OnsetDelay_SingleBlock = 0.400;
                S.GUI.AirPuff_OnsetDelay_Short = 0.200;
                S.GUI.AirPuff_OnsetDelay_Long  = 0.400;
                S.GUI.num_warmup_trials = 15;
                S.GUI.CheckEyeOpenAveragingBaseline = 0.2;
                S.GUI.CheckEyeOpenTimeout = 15;
                % ------------------------
                % Training stage (still available but probes won't depend on it)
                % ------------------------
                S.GUI.SleepDeprived = 1;
                S.GUIMeta.SleepDeprived.Style = 'popupmenu';
                S.GUIMeta.SleepDeprived.String = {'Control','Pre_EBC_SD'};
                S.GUI.TrainingStage = 1;
                S.GUIMeta.TrainingStage.Style = 'popupmenu';
                S.GUIMeta.TrainingStage.String = {'naive','well-trained'};



                % ------------------------
                % Probe Trials Parameters
                % ------------------------
                S.GUI.UseProbeTrials = 1;                                % independent from training stage
                S.GUIMeta.UseProbeTrials.Style = 'checkbox';
                S.GUIMeta.UseProbeTrials.String = '(check to activate)';
                S.GUI.probetrials_percentage_perBlock = 0;               % % of eligible trials
                S.GUIMeta.probetrials_percentage_perBlock.Style = 'edit';
                S.GUIMeta.probetrials_percentage_perBlock.String = '% of block (excl. lead-in/last)';
                S.GUI.num_initial_nonprobe_trials_perBlock = 7;          % lead-in per block
                S.GUIMeta.num_initial_nonprobe_trials_perBlock.Style = 'edit';
                S.GUIMeta.num_initial_nonprobe_trials_perBlock.String = 'lead-in trials';
                % spacing rule (>=3 non-probes between probes)
                S.GUI.ProbeMinSeparation = 3;                             % distance in trials
                S.GUIMeta.ProbeMinSeparation.Style = 'edit';
                S.GUIMeta.ProbeMinSeparation.String = 'min spacing (≥4 ⇒ 3 gaps)';
                % ------------------------
                % Optogenetic parameters (Organized)
                % ------------------------
                % === Common to all session types ===
                S.GUI.OptoEnabled = 1;
                S.GUIMeta.OptoEnabled.Style = 'checkbox';
                S.GUI.OptoSessionType = 1;
                S.GUIMeta.OptoSessionType.Style = 'popupmenu';
                S.GUIMeta.OptoSessionType.String = {'RandomTrial', 'ProbeTrialOpto', 'BlockTransition'};
                S.GUI.OptoOnset = 2;  % popup index
                S.GUIMeta.OptoOnset.Style = 'popupmenu';
                S.GUIMeta.OptoOnset.String = {'Same as Airpuff', '0 ms', '200 ms', '400 ms'};
                S.GUI.OptoDuration = 0.250;  % legacy fallback; specific fields below are used per block type
                S.GUI.OptoDuration_SingleBlock = 0.250;
                S.GUIMeta.OptoDuration_SingleBlock.Style = 'edit';
                S.GUIMeta.OptoDuration_SingleBlock.String = 'single block duration (s)';
                S.GUI.OptoDuration_DoubleBlockShort = 0.250;
                S.GUIMeta.OptoDuration_DoubleBlockShort.Style = 'edit';
                S.GUIMeta.OptoDuration_DoubleBlockShort.String = 'double block short duration (s)';
                S.GUI.OptoDuration_DoubleBlockLong = 0.450;
                S.GUIMeta.OptoDuration_DoubleBlockLong.Style = 'edit';
                S.GUIMeta.OptoDuration_DoubleBlockLong.String = 'double block long duration (s)';
                S.GUI.OptoPattern = 1;
                S.GUIMeta.OptoPattern.Style = 'popupmenu';
                S.GUIMeta.OptoPattern.String = {'HoldAndRampDown', 'Ramp', 'Hold'};
                % === Random Trials specific ===
                S.GUI.OptoFraction = 0.2;
                % === Probe Trials specific ===
                S.GUI.OptoFractionProbe = 1.0;
                % === Block Transition specific ===
                S.GUI.BlockTypeReceiveOpto = 1;
                S.GUIMeta.BlockTypeReceiveOpto.Style = 'popupmenu';
                S.GUIMeta.BlockTypeReceiveOpto.String = {'LongBlockOnly', 'ShortBlockOnly', 'BothBlocksShortLong'};
                S.GUI.WhichBlockTransitions = 2;
                S.GUIMeta.WhichBlockTransitions.Style = 'popupmenu';
                S.GUIMeta.WhichBlockTransitions.String = {'AllTransitions', 'AlternatingTransitions'};
                S.GUI.OptoMinSeparation = 3;
                S.GUI.OptoInitialTrials = 5;
                
                % ------------------------
                % Chemogenetic parameters 
                % ------------------------
                % === Common to all session types ===
                S.GUI.ChemogeneticsEnabled = 0;
                S.GUIMeta.ChemogeneticsEnabled.Style = 'checkbox';
                % ------------------------
                % GUI Panels (boxes)
                % ------------------------
                S.GUIPanels.EBC = { ...
                    'ExperimenterInitials','WheelDirection','WheelRadius','SleepDeprived','TrainingStage', ...
                    'num_warmup_trials','ITI_Pre','ITI_Post','ITI_Extra', ...
                    'TrialTypeSequence','LED_Dur',  ...
                    'AirPuff_OnsetDelay_SingleBlock','AirPuff_OnsetDelay_Short','AirPuff_OnsetDelay_Long', ...
                    'AirPuff_Dur','CheckEyeOpenAveragingBaseline','CheckEyeOpenTimeout', ...
                    'BlockLength','Margine'};
                % Probes in their own panel
                S.GUIPanels.ProbeTrials = { ...
                    'UseProbeTrials', ...
                    'probetrials_percentage_perBlock', ...
                    'num_initial_nonprobe_trials_perBlock', ...
                    'ProbeMinSeparation'};
                % Optogenetics panel (updated to match new fields)
                S.GUIPanels.Optogenetics = { ...
                    'OptoEnabled','OptoSessionType','OptoOnset', ...
                    'OptoDuration_SingleBlock','OptoDuration_DoubleBlockShort','OptoDuration_DoubleBlockLong', ...
                    'OptoPattern', ...
                    'OptoFraction','OptoFractionProbe','BlockTypeReceiveOpto','WhichBlockTransitions',...
                    'OptoMinSeparation','OptoInitialTrials'};
                % Chemoogenetics panel
                S.GUIPanels.Chemoogenetics = {'ChemogeneticsEnabled' };

                % Airpuff test button
                S.GUI.ActivateAirPuffPulse = @(src,event)fnActivateAirPuffPulse;
                S.GUIMeta.ActivateAirPuffPulse.Style = 'pushbutton';
                S.GUI.AirPuff_Pulse_Dur = 0.02;
                S.GUIPanels.AirPuffPulse = {'ActivateAirPuffPulse','AirPuff_Pulse_Dur'};

                % imaging rig pupil color
                S.GUI.PupilColorForImaging = 1;
                S.GUIMeta.PupilColorForImaging.Style = 'popupmenu';
                S.GUIMeta.PupilColorForImaging.String = {'Black','White'};
                S.GUIPanels.Imaging = { 'PupilColorForImaging'};
            end

            % Opto-session preset: random opto, no probe trials, LED-anchored opto.
            S.GUI.probetrials_percentage_perBlock = 0;
            S.GUI.OptoSessionType = 1;
            S.GUI.OptoOnset = 2;
            S.GUI.OptoDuration = 0.250;
            S.GUI.OptoDuration_SingleBlock = 0.250;
            S.GUI.OptoDuration_DoubleBlockShort = 0.250;
            S.GUI.OptoDuration_DoubleBlockLong = 0.450;
            if ~isfield(S.GUI, 'OptoInitialTrials')
                S.GUI.OptoInitialTrials = 5;
            end

            if ~isfield(S.GUI, 'OptoDuration')
                S.GUI.OptoDuration = 0.1;
            end
            if ~isfield(S.GUI, 'OptoDuration_SingleBlock')
                S.GUI.OptoDuration_SingleBlock = S.GUI.OptoDuration;
            end
            S.GUIMeta.OptoDuration_SingleBlock.Style = 'edit';
            S.GUIMeta.OptoDuration_SingleBlock.String = 'single block duration (s)';
            if ~isfield(S.GUI, 'OptoDuration_DoubleBlockShort')
                S.GUI.OptoDuration_DoubleBlockShort = S.GUI.OptoDuration;
            end
            S.GUIMeta.OptoDuration_DoubleBlockShort.Style = 'edit';
            S.GUIMeta.OptoDuration_DoubleBlockShort.String = 'double block short duration (s)';
            if ~isfield(S.GUI, 'OptoDuration_DoubleBlockLong')
                S.GUI.OptoDuration_DoubleBlockLong = S.GUI.OptoDuration;
            end
            S.GUIMeta.OptoDuration_DoubleBlockLong.Style = 'edit';
            S.GUIMeta.OptoDuration_DoubleBlockLong.String = 'double block long duration (s)';
            S.GUIPanels.Optogenetics = { ...
                'OptoEnabled','OptoSessionType','OptoOnset', ...
                'OptoDuration_SingleBlock','OptoDuration_DoubleBlockShort','OptoDuration_DoubleBlockLong', ...
                'OptoPattern', ...
                'OptoFraction','OptoFractionProbe','BlockTypeReceiveOpto','WhichBlockTransitions',...
                'OptoMinSeparation','OptoInitialTrials'};
        end
    end
end

