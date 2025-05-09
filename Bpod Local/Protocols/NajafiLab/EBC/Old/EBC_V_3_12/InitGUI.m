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
        S.GUI.LED_Dur = 0.5;
        S.GUI.LED_Dur_Short = 0.5;
        S.GUI.LED_Dur_Long = 0.5;

        S.GUI.BlockLength = 50;
        S.GUI.Margine = 5;

        S.GUI.TrialTypeSequence = 2;   % trial type sequence selection
        % 1 - Normal - AirPuff_OnsetDelay is calculated to be at the end of the LED duration 
        % 2 - Short/Long - AirPuff_OnsetDelay uses the short or long onset delay values depending upon the current trial type (short or long)
        S.GUIMeta.TrialTypeSequence.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.TrialTypeSequence.String = {'Normal', 'Short/Long'};  
        
        S.GUI.AirPuff_Dur = 0.02;
        S.GUI.AirPuff_OnsetDelay = S.GUI.LED_Dur - S.GUI.AirPuff_Dur;
        

        S.GUI.AirPuff_OnsetDelay_Short = 0.200;
        S.GUI.AirPuff_OnsetDelay_Long = 0.400;
        
        S.GUI.CheckEyeOpenAveragingBaseline = 0.2;
        S.GUI.CheckEyeOpenTimeout = 15;

        S.GUI.SleepDeprived = false;
        S.GUIMeta.SleepDeprived.Style = 'checkbox';

        S.GUIPanels.EBC = {'ExperimenterInitials','SleepDeprived', 'ITI_Pre', 'ITI_Post', 'ITI_Extra','LED_OnsetDelay',...
            'TrialTypeSequence','LED_Dur_Short','LED_Dur_Long', 'AirPuff_OnsetDelay', 'AirPuff_OnsetDelay_Short','AirPuff_OnsetDelay_Long', 'AirPuff_Dur', 'CheckEyeOpenAveragingBaseline','CheckEyeOpenTimeout','BlockLength','Margine'};

    end
end




    end
end
