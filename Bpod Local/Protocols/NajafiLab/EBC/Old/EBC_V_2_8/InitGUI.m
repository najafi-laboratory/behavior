classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S

    if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
        
        
        S.GUI.ITI_Pre = 0.5;
        S.GUI.ITI_Post = 5;

        S.GUI.LED_OnsetDelay = 0;
        S.GUI.LED_Dur = 0.5;

        S.GUI.AirPuff_Dur = 0.02;
        S.GUI.AirPuff_OnsetDelay = S.GUI.LED_Dur - S.GUI.AirPuff_Dur;
        
        

        S.GUIPanels.EBC = {'ITI_Pre', 'ITI_Post', 'LED_OnsetDelay', 'LED_Dur', 'AirPuff_OnsetDelay', 'AirPuff_Dur'};

    end
end















    end
end
