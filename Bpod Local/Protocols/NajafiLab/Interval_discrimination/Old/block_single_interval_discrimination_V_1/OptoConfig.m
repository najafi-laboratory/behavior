classdef OptoConfig

properties
    laser_module = [];
end

methods


function obj = ConnectInitLaser(obj, S)
    if S.GUI.EnDoricAPI
        pythonDoricPath = 'C:\DoricLaserAPI\DoricSystemDLL\Examples\Python\LightSource';
        insert(py.sys.path, int32(0), pythonDoricPath);  % Add Python script location
        % clear classes;
        obj.laser_module = py.importlib.import_module('light_source_main'); % Import the Python script
        py.importlib.reload(obj.laser_module);
        result = obj.laser_module.InitLaser();   % Call the Python function
        disp(result);
    end
end


function obj = DisconnectLaser(obj, S)
    if S.GUI.EnDoricAPI
        result = obj.laser_module.DisconnectLaser();
        disp(result);
    end
end     


function [OptoType] = GenOptoType(obj, S, BlockTypes)
    if (S.GUI.OptoSession == 1)
        OptoType = zeros(1, S.GUI.MaxTrials);
    elseif (S.GUI.OptoSession == 2)
        OptoType = 0 + (rand(1, S.GUI.MaxTrials) < S.GUI.OnFraction);
        OptoType(1:S.GUI.EarliestOpto) = 0;
    elseif (S.GUI.OptoSession == 3)
        OptoType = zeros(1, numel(BlockTypes));
        edges = [1 find(diff(BlockTypes))+1 numel(BlockTypes)+1];
        for i = 1:numel(edges)-1
            idx = edges(i):edges(i+1)-1;
            groupFlag = mod(floor(i/S.GUI.NumBlockType),2);
            nTrial = min(S.GUI.OptoEarlyNumTrial, numel(idx));
            OptoType(idx(1:nTrial)) = (rand(1,nTrial) < S.GUI.OptoEarlyProb) .* (1 - groupFlag);
            OptoType(1:S.GUI.EarliestOpto) = 0;
        end
    elseif (S.GUI.OptoSession == 4)
        OptoType = zeros(1, numel(BlockTypes));
        edges = [1 find(diff(BlockTypes))+1 numel(BlockTypes)+1];
        for i = 1:numel(edges)-1
            idx = edges(i):edges(i+1)-1;
            nTrial = min(S.GUI.OptoEarlyNumTrial, numel(idx));
            OptoType(idx(1:nTrial)) = (rand(1,nTrial) < S.GUI.OptoEarlyProb);
            OptoType(1:S.GUI.EarliestOpto) = 0;
        end
    end

    OptoType = AssignEnabledOptoTypes(obj, S, OptoType);
end


function [OptoType] = AssignEnabledOptoTypes(obj, S, OptoType)
    enabledTypes = GetEnabledOptoTypes(obj, S);
    optoTrials = find(OptoType > 0);
    if isempty(enabledTypes)
        OptoType(optoTrials) = 0;
        return
    end
    OptoType(optoTrials) = enabledTypes(randi(numel(enabledTypes), 1, numel(optoTrials)));
end


function [enabledTypes] = GetEnabledOptoTypes(obj, S)
    enableType1 = ~isfield(S.GUI, 'EnableOptoType1') || S.GUI.EnableOptoType1 == 1;
    enableType2 = isfield(S.GUI, 'EnableOptoType2') && S.GUI.EnableOptoType2 == 1;
    enabledTypes = [ones(1, enableType1), 2*ones(1, enableType2)];
end

function [AudStim] = OptoUpdateAudStimTrig(obj, OptoType, currentTrial)
    switch OptoType(currentTrial)
        case 0
            AudStim = {'HiFi1', ['P', 4]};
        case 1
            AudStim = {'HiFi1', ['P', 4], 'GlobalTimerTrig', '11'};
        case 2
            AudStim = {'HiFi1', ['P', 4], 'GlobalTimerTrig', '1100'};
    end
end


function [sma] = SetOpto(obj, BpodSystem, S, sma, OptoDuration, OptoType, currentTrial)
    switch OptoType(currentTrial)
        case 1
            sma = SetPulseTrainTimers(obj, S, sma, OptoDuration, 1, 2);
        case 2
            sma = SetPulseTrainTimers(obj, S, sma, OptoDuration, 3, 4);
    end
end


function [sma] = SetPulseTrainTimers(obj, S, sma, OptoDuration, ledTimerID, pmtTimerID)
    PMTStartCloseDelay = 0.010;
    PMTCloseTransferDelay = 0.0121;
    PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;
    PMTMin5VSignalDur = PMTCloseDelay + 0.003;
    PMTStartOpenDelay = 0.0078;
    LEDOnsetDelay = 0;
    PMTOnsetDelay = LEDOnsetDelay - PMTCloseDelay;
    LEDOffDur = S.GUI.OptoFreq - S.GUI.LEDOnPulseDur;
    PMT5VDur = PMTCloseDelay;
    if S.GUI.LEDOnPulseDur > PMTStartOpenDelay
        PMT5VDur = PMT5VDur + (S.GUI.LEDOnPulseDur - PMTStartOpenDelay);
    end
    PMT5VDur = max(PMT5VDur, PMTMin5VSignalDur);
    PMT0VDur = S.GUI.OptoFreq - PMT5VDur;
    numPMTLEDCycles = floor(OptoDuration / S.GUI.OptoFreq);
    sma = SetGlobalTimer(sma, 'TimerID', ledTimerID, 'Duration', S.GUI.LEDOnPulseDur, 'OnsetDelay', LEDOnsetDelay,...
        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
        'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
        'GlobalTimerEvents', 0, 'OffsetValue', 0);
    sma = SetGlobalTimer(sma, 'TimerID', pmtTimerID, 'Duration', PMT5VDur, 'OnsetDelay', PMTOnsetDelay,...
        'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
        'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMT0VDur,...
        'GlobalTimerEvents', 0, 'OffsetValue', 0);
end


function [OptoDuration] = GetOptoDuration(obj, S, VisStim, DURA, OptoType, currentTrial)
    switch OptoType(currentTrial)
        case 1
            OptoDuration = VisStim.Data.Post.Dur;
        case 2
            OptoDuration = VisStim.Data.VisStimDuration + DURA.PostVisStimDelay + ...
                S.GUI.PreGoCueDelay_s + S.GUI.GoCueDuration_s + DURA.ChoiceWindow + ...
                S.GUI.OutcomeFeedbackDelay + S.GUI.PreRewardDelay + ...
                max([S.GUI.PostRewardDelay, S.GUI.PostPunishDelay, S.GUI.PostOutcomeDelay]) + ...
                DURA.TimeOutPunish + 5;
        otherwise
            OptoDuration = 0;
    end
end


function [ITIAction] = GetITIAction(obj, OptoType, currentTrial)
    if OptoType(currentTrial) == 2
        ITIAction = {'GlobalTimerCancel', '1100'};
    else
        ITIAction = {};
    end
end



    end
end