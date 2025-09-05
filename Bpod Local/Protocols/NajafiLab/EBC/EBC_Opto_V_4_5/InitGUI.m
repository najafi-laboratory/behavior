classdef InitGUI
    methods

    function [S] = SetParams(obj, BpodSystem)
        S = BpodSystem.ProtocolSettings;
        if isempty(fieldnames(S))  % Default settings
            S.GUI.ExperimenterInitials = 'Initials';
            S.GUI.ITI_Pre = 1;
            S.GUI.ITI_Post = 3;
            S.GUI.ITI_Extra = 3;
            S.GUI.LED_OnsetDelay = 0;
            S.GUI.LED_Dur = 0.05;
            S.GUI.BlockLength = 50;
            S.GUI.Margine = 5;
            S.GUI.TrialTypeSequence = 2;
            S.GUIMeta.TrialTypeSequence.Style = 'popupmenu';
            S.GUIMeta.TrialTypeSequence.String = {'singleBlock','singleTransition_short_to_long','singleTransition_long_to_short','doubleBlock_shortFirst','doubleBlock_longFirst','doubleBlock_RandomFirst'};  
            S.GUI.AirPuff_Dur = 0.02;
            S.GUI.AirPuff_OnsetDelay_SingleBlock = 0.400;
            S.GUI.AirPuff_OnsetDelay_Short = 0.200;
            S.GUI.AirPuff_OnsetDelay_Long = 0.400;
            S.GUI.num_warmup_trials = 15;
            S.GUI.CheckEyeOpenAveragingBaseline = 0.2;
            S.GUI.CheckEyeOpenTimeout = 15;
            S.GUI.SleepDeprived = 1;
            S.GUIMeta.SleepDeprived.Style = 'popupmenu';
            S.GUIMeta.SleepDeprived.String = {'Control','Pre_EBC_SD'};
            S.GUI.probetrials_percentage_perBlock = 15;
            S.GUI.num_initial_nonprobe_trials_perBlock = 5;
            S.GUI.TrainingStage = 1;
            S.GUIMeta.TrainingStage.Style = 'popupmenu';
            S.GUIMeta.TrainingStage.String = {'naive', 'well-trained'};
            % ------------------------
            % Optogenetic parameters (Organized)
            % ------------------------
            
            % === Common to all session types ===
            S.GUI.OptoEnabled = 1;
            S.GUIMeta.OptoEnabled.Style = 'checkbox';
            
            S.GUI.OptoSessionType = 1;
            S.GUIMeta.OptoSessionType.Style = 'popupmenu';
            S.GUIMeta.OptoSessionType.String = {'RandomTrial', 'ProbeTrialOpto', 'BlockTransition'};
            
            S.GUI.OptoOnset = 1;
            S.GUIMeta.OptoOnset.Style = 'popupmenu';
            S.GUIMeta.OptoOnset.String = {'Same as Airpuff', '0 ms', '200 ms', '400 ms'};
            
            S.GUI.OptoDuration = 0.1;
            
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
            
            % === GUI Panels ===
            S.GUIPanels.OptoCommon = {'OptoEnabled', 'OptoSessionType', 'OptoOnset', 'OptoDuration', 'OptoPattern'};
            S.GUIPanels.OptoRandomPanel = {'OptoFraction'};
            S.GUIPanels.OptoProbePanel = {'OptoFractionProbe'};
            S.GUIPanels.OptoBlockTransitionPanel = {'BlockTypeReceiveOpto', 'WhichBlockTransitions'};
            
            % GUI Panels
            S.GUIPanels.EBC = {'ExperimenterInitials', 'SleepDeprived','TrainingStage', ...
                'probetrials_percentage_perBlock', 'num_initial_nonprobe_trials_perBlock', ...
                'num_warmup_trials', 'ITI_Pre', 'ITI_Post', 'ITI_Extra', ...
                'TrialTypeSequence', 'LED_Dur', ...
                'AirPuff_OnsetDelay_SingleBlock', 'AirPuff_OnsetDelay_Short', 'AirPuff_OnsetDelay_Long', ...
                'AirPuff_Dur', 'CheckEyeOpenAveragingBaseline', 'CheckEyeOpenTimeout', ...
                'BlockLength', 'Margine'};
            % Airpuff test button
            S.GUI.ActivateAirPuffPulse = @(src,event)fnActivateAirPuffPulse;
            S.GUIMeta.ActivateAirPuffPulse.Style = 'pushbutton';
            S.GUI.AirPuff_Pulse_Dur = 0.02;
            S.GUIPanels.AirPuffPulse = {'ActivateAirPuffPulse', 'AirPuff_Pulse_Dur'};
        end
    end


    end
end
