classdef TrialConfig
    methods


%% trial generation

function [TrialTypes] = GenTrials(obj, S)
    TrialTypes = ceil(rand(1, 1106)*2); 
end


%% random isi


function [JitterFlag] = GetJitterFlag( ...
        obj, S)
    if (S.GUI.ActRandomISI == 1)
        JitterFlag = 1;
    else
        JitterFlag = 0;
    end
end


%% anti-bias algorithm

% get valve time from valve amount
function [ValveTime] = Amount2Time( ...
        obj, Amount, Valve)
    ValveTime = GetValveTimes(Amount, Valve);
    if (ValveTime < 0)
        ValveTime = 0;
    end
end


%% ITI and timeout ITI


function [ITI] = GetITI( ...
        obj, S)
    ITI = 19961106;
    if (S.GUI.SetManualITI == 1)
        ITI = S.GUI.ManualITI;
    else
        while (ITI < S.GUI.ITIMin || ITI > S.GUI.ITIMax)
            ITI = -log(rand) * S.GUI.ITIMean;
        end
    end
end


function [TimeOutPunish] = GetTimeOutPunish( ...
        obj, S)
    if (S.GUI.ActTimeOutPunish == 1)
        if (S.GUI.ManuallTimeOutPunish)
            TimeOutPunish = S.GUI.TimeOutPunish;
        else
            switch S.GUI.TrainingLevel
                case 1
                    TimeOutPunish = 1.5;
            end
        end
    else
        TimeOutPunish = 0;
    end
end


%% choice window

function [ChoiceWindow] = GetChoiceWindow( ...
        obj, S)
    if (S.GUI.ManualChoiceWindow)
        ChoiceWindow = S.GUI.ChoiceWindow_s;
    else
        switch S.GUI.TrainingLevel
            case 1
                ChoiceWindow = 15;
        end
    end
end



%% moving spouts


function SetMotorPos = ConvertMaestroPos(obj, MaestroPosition)
    m = 0.002;
    b = -3;
    SetMotorPos = MaestroPosition * m + b;
end

































    end
end