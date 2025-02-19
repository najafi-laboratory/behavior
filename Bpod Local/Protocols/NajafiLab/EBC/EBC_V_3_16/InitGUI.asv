classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S

    if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
        
        S.GUI.ExperimenterInitials = 'Initials';
        

        S.GUI.ITI_Pre = 0.5;
        S.GUI.ITI_Post = 5;
        S.GUI.ITI_Extra = 3;

        S.GUI.LED_OnsetDelay = 0;

        S.GUI.LED_Dur = 0.05;

        S.GUI.BlockLength = 50;
        S.GUI.Margine = 5;

        % Trial Type Sequence selection (Dropdown menu)
        S.GUI.TrialTypeSequence = 2;  % Default: 'doubleBlock_shortFirst'
        % 1 - singleBlock 
        % 2 - doubleBlock_shortFirst - start with short blocks
        % 3 - doubleBlock_longFirst -  starts with long blocks 
        % 4 - doubleBlock_RandomFirst - starts randomly with short or long blocks 
        S.GUIMeta.TrialTypeSequence.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.TrialTypeSequence.String = {'singleBlock','doubleBlock_shortFirst','doubleBlock_longFirst','doubleBlock_RandomFirst'};  
                
        S.GUI.AirPuff_Dur = 0.02;

        S.GUI.AirPuff_OnsetDelay = S.GUI.LED_Dur - S.GUI.AirPuff_Dur;       

        S.GUI.AirPuff_OnsetDelay_SingleBlock = 0.400; % Default ISI for single block
        S.GUI.AirPuff_OnsetDelay_Short = 0.200;
        S.GUI.AirPuff_OnsetDelay_Long = 0.400;

        % Warm-up trials
        S.GUI.num_warmup_trials = 15;  % Default 15 warm-up trials

        
        S.GUI.CheckEyeOpenAveragingBaseline = 0.2;
        S.GUI.CheckEyeOpenTimeout = 15;

        S.GUI.SleepDeprived = false;
        S.GUIMeta.SleepDeprived.Style = 'checkbox';
        
        % GUI Panels
        S.GUIPanels.EBC = {'ExperimenterInitials', 'SleepDeprived', 'num_warmup_trials', 'ITI_Pre', 'ITI_Post', 'ITI_Extra', ...
            'TrialTypeSequence', 'LED_Dur', 'AirPuff_OnsetDelay', ...
            'AirPuff_OnsetDelay_SingleBlock', 'AirPuff_OnsetDelay_Short', 'AirPuff_OnsetDelay_Long', ...
            'AirPuff_Dur', 'CheckEyeOpenAveragingBaseline', 'CheckEyeOpenTimeout', 'BlockLength', 'Margine'};

        % Define a button for activating the airpuff pulse
        S.GUI.ActivateAirPuffPulse = @(src,event)fnActivateAirPuffPulse;
        S.GUIMeta.ActivateAirPuffPulse.Style = 'pushbutton';
        S.GUI.AirPuff_Pulse_Dur = 0.02; % Default amount as specified
        S.GUIPanels.AirPuffPulse = {'ActivateAirPuffPulse', 'AirPuff_Pulse_Dur'};


    end
end


    end
end
