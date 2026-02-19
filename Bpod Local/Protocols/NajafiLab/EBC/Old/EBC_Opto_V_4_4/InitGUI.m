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
            % Optogenetic parameters
            % ------------------------
            S.GUI.OptoEnabled = 1;
            S.GUIMeta.OptoEnabled.Style = 'checkbox';
            S.GUI.OptoSessionType = 1;
            S.GUIMeta.OptoSessionType.Style = 'popupmenu';
            % S.GUIMeta.OptoSessionType.String = {'RandomTrial', 'BlockTransition'};
            S.GUIMeta.OptoSessionType.String = {'RandomTrial', 'BlockTransition', 'ProbeTrialOpto'};
            % For both session types
            S.GUI.OptoOnset = 0.4;  % default = AirPuff_OnsetDelay
            S.GUI.OptoDuration = 0.1;
            S.GUI.OptoPattern = 1;
            S.GUIMeta.OptoPattern.Style = 'popupmenu';
            S.GUIMeta.OptoPattern.String = {'HoldAndRampDown', 'SquarePulse'};
            % For Random Trial opto (SessionType = 1)
            S.GUI.OptoFraction = 0.2;
            % For Block Transition opto (SessionType = 2)
            S.GUI.OptoTransitionTrials = 5;
            S.GUI.OptoTransitionType = 1;
            S.GUIMeta.OptoTransitionType.Style = 'popupmenu';
            S.GUIMeta.OptoTransitionType.String = {'AlternatingShortToLong', 'AllShortToLong'};
            S.GUIPanels.Optogenetics = {'OptoEnabled', 'OptoSessionType', 'OptoOnset', 'OptoDuration', 'OptoPattern', ...
                                        'OptoFraction', 'OptoTransitionTrials', 'OptoTransitionType'};
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
