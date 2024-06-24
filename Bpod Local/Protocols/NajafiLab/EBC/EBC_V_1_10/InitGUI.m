classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S

    if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
        
        
        S.GUI.ITI_Pre = 0.5;
        S.GUI.ITI_Post = 1;

        S.GUI.LED_OnsetDelay = 0;
        S.GUI.LED_Dur = 4;

        S.GUI.AirPuff_OnsetDelay = 1;
        S.GUI.AirPuff_Dur = 2;

        S.GUIPanels.EBC = {'ITI_Pre', 'ITI_Post', 'LED_OnsetDelay', 'LED_Dur', 'AirPuff_OnsetDelay', 'AirPuff_Dur'};

        S.GUI.NumPulses = 3;
        S.GUI.PulseDur = 0.250;
        S.GUI.IPI = 0.5;
        S.GUI.currentTrial = 0;
        S.GUIPanels.Opto = {'NumPulses', 'PulseDur', 'IPI', 'currentTrial'};
    end
end















    end
end
