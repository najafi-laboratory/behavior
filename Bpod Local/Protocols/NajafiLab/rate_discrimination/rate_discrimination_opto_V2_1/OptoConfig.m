classdef OptoConfig

properties
    EnableOpto = 0;
end

methods

function obj = OptoConfig(EnableOpto)
    if nargin == 1
        obj.EnableOpto = EnableOpto;
    end
end

function obj = set.EnableOpto(obj, EnableOpto)
    obj.EnableOpto = EnableOpto;
end

function [OptoTrialTypes] = GenOptoTrials(obj, BpodSystem, S)
    BpodSystem.Data.PreviousSessionType = S.GUI.SessionType;
    BpodSystem.Data.PreviousOptoTrialTypeSeq = S.GUI.OptoTrialTypeSeq;
    BpodSystem.Data.PreviousOnFraction = S.GUI.OnFraction;
    BpodSystem.Data.PreviousNumOptoTrialsPerBlock = S.GUI.NumOptoTrialsPerBlock;
    OptoTrialTypes = ceil(rand(1, S.GUI.MaxTrials)*2);
end

function [OptoTrialTypes] = UpdateOptoTrials(obj, BpodSystem, S, OptoTrialTypes, currentTrial, forceUpdate)
    updateOptoTrialTypeSequence = 0;

    if forceUpdate == 1
        updateOptoTrialTypeSequence = 1;
    end

    if S.GUI.SessionType == 2
            S.GUI.OptoTrialTypeSeq = 1;
            S.GUI.OnFraction = 0;
            updateOptoTrialTypeSequence = 1;
    end

    if (BpodSystem.Data.PreviousSessionType ~= S.GUI.SessionType) && (S.GUI.SessionType == 1)
        updateOptoTrialTypeSequence = 1;           
    end
    BpodSystem.Data.PreviousSessionType = S.GUI.SessionType;

    if BpodSystem.Data.PreviousOptoTrialTypeSeq ~= S.GUI.OptoTrialTypeSeq
        updateOptoTrialTypeSequence = 1;
    end
    BpodSystem.Data.PreviousOptoTrialTypeSeq = S.GUI.OptoTrialTypeSeq;

    if BpodSystem.Data.PreviousNumOptoTrialsPerBlock ~= S.GUI.NumOptoTrialsPerBlock
        updateOptoTrialTypeSequence = 1;
        BpodSystem.Data.PreviousNumOptoTrialsPerBlock = S.GUI.NumOptoTrialsPerBlock;
    end

    if BpodSystem.Data.PreviousOnFraction ~= S.GUI.OnFraction
        updateOptoTrialTypeSequence = 1;
        BpodSystem.Data.PreviousOnFraction = S.GUI.OnFraction;
    end

    if updateOptoTrialTypeSequence
        numTrialsAddedToSequence = 0;
        OptoTrialTypesToAdd = [];
        switch S.GUI.OptoTrialTypeSeq
            case 1 % interleaved - random
                numOnTrialsToAdd = [];
                OptoTrialTypesToAdd = [repmat(1, 1, S.GUI.MaxTrials - currentTrial + 1)];
                numOnTrialsToAdd = ceil(S.GUI.OnFraction * length(OptoTrialTypesToAdd));
                numOnTrialsToAdd_idxs = randperm(length(OptoTrialTypesToAdd), numOnTrialsToAdd);
                OptoTrialTypesToAdd(numOnTrialsToAdd_idxs) = 2;
                OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd;
            case 2 % interleaved - random first trial type
                FirstTrialType = ceil(rand(1,1)*2);
                switch FirstTrialType
                    case 1 % on first
                        SecondTrialType = 2;
                    case 2 % off first
                        SecondTrialType = 1;
                end
                while numTrialsAddedToSequence < S.GUI.MaxTrials
                    OptoTrialTypesToAdd = [OptoTrialTypesToAdd repmat(FirstTrialType, 1, S.GUI.NumOptoTrialsPerBlock) repmat(SecondTrialType, 1, S.GUI.NumOptoTrialsPerBlock)];
                    numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumOptoTrialsPerBlock;
                end
                OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd(1:length(OptoTrialTypes) - currentTrial + 1);
            case 3 % interleaved - short first block
                while numTrialsAddedToSequence < S.GUI.MaxTrials
                    OptoTrialTypesToAdd = [OptoTrialTypesToAdd repmat(1, 1, S.GUI.NumOptoTrialsPerBlock) repmat(2, 1, S.GUI.NumOptoTrialsPerBlock)];
                    numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumOptoTrialsPerBlock;
                end
                OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd(1:length(OptoTrialTypes) - currentTrial + 1);
            case 4 % interleaved - long first block
                while numTrialsAddedToSequence < S.GUI.MaxTrials
                    OptoTrialTypesToAdd = [OptoTrialTypesToAdd repmat(2, 1, S.GUI.NumOptoTrialsPerBlock) repmat(1, 1, S.GUI.NumOptoTrialsPerBlock)];
                    numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumOptoTrialsPerBlock;
                end
                OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd(1:length(OptoTrialTypes) - currentTrial + 1);
        end
    end
end

function [AudStimOpto] = GetAudStimOpto(obj, OptoTrialType)
    switch obj.EnableOpto
        case 0
            AudStimOpto = {'HiFi1', ['P', 4]};
        case 1
            if OptoTrialType == 2
                AudStimOpto = {'HiFi1', ['P', 4], 'GlobalTimerTrig', '11'};
            else
                AudStimOpto = {'HiFi1', ['P', 4]};
            end
    end
end

function [sma] = SetOpto(obj, BpodSystem, S, sma, VisStim, OptoTypes, currentTrial)
    if OptoTypes(currentTrial)
% VisStim.Data.VisStimDuration
        % initial gray frame vis stim offset, statistical delay of
        % 2 frames at 60fps
        VisStimShift = 0;
        switch BpodSystem.Data.RigName
            case 'ImagingRig'
                % % measured gray f1, f2 duration
                % VisStimShift = 0.032292;
                VisStimShift = 0.0329645; % f1 + f2 - imaging rig              
            case 'JoystickRig'
                % VisStimShift = 0.031698; % f1 + f2 - joystick rig
                f1 = 0.0153655; % gray f1 dur
                f2 = 0.0176069; % gray f2 dur
                VisStimShift = f1 + f2; % f1 + f2 - joystick rig
            case 'Rig2'
                % % measured gray f1, f2 duration
                VisStimShift = 0.0325; % f1 + f2 - imaging rig
        end


        % pmt shutter delay timings are from Bruker specs, the actual
        % timings are a bit different, and vary between green/red PMTs, but these work for pulsing opto
        % shutter close delays
        PMTStartCloseDelay = 0.010;     % ~10ms for shutter to start closing after setting shutter signal to 5V
        PMTCloseTransferDelay = 0.0121; % ~12.1ms
        PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;          
        PMTMin5VSignalDur = PMTCloseDelay + 0.003; % ~25.1ms, minimum measured duration of 5V shutter signal for shutter to re-open 
        % shutter open delays
        PMTStartOpenDelay = 0.0078;     % ~7.8ms for shutter to start opening after setting shutter signal to 0V
        PMTOpenTransferDelay = 0.0125;  % ~12.5ms for shutter to open after the 7.8ms open transfer delay
        % opto cycle timing is defined by the onset of the LED
        % paddedISIseq = [ISIseq(1), ISIseq];
        % LEDOnsetDelay = paddedISIseq(currentTrial)/2 - S.GUI.OptoPreVisOnset;
        % LEDOnsetDelay = VisStim.Data.OptoGrayDur;
        LEDOnsetDelay = VisStimShift;

        PMTOnsetDelay = LEDOnsetDelay - PMTCloseDelay;
        % 10Hz pulsed shutter/opto, 7.8ms LED
        LEDOnDur = S.GUI.LEDOnPulseDur_ms/1000;
        LEDOffDur = S.GUI.LEDOffPulseDur_ms/1000;
        T = LEDOnDur + LEDOffDur;
        % PMT shutter signal 5V and 0V durations           
        PMT5VDur = PMTCloseDelay; % set PMT5V dur initially to shutter close delay
        % if the LED is on for longer than the shutter StartOpenDelay,
        % then increase shutter 5V duration by the difference (LEDOnDur - PMTStartOpenDelay)
        if LEDOnDur > PMTStartOpenDelay
            PMT5VDur = PMT5VDur + (LEDOnDur - PMTStartOpenDelay);
        end
        % if shutter duration is less than the minimum dur for the
        % shutter to re-open, set it to minimum shutter pulse dur
        PMT5VDur = max(PMT5VDur, PMTMin5VSignalDur);
        % duration of LED pulse 0V is cycle period minus 5V dur
        PMT0VDur =  T - PMT5VDur;
        % integer number of pmt/led cycles within pre + post grating onset
        % numPMTLEDCycles = floor((S.GUI.OptoPreVisOnset + S.GUI.GratingDur_s + S.GUI.OptoPostVisOnset) / S.GUI.OptoFreq);     
        numPMTLEDCycles = min(S.GUI.MaxOptoDur_s, VisStim.Data.VisStimDuration)/T;

        % LED timers
        sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', LEDOnDur, 'OnsetDelay', LEDOnsetDelay,...
            'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
            'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
            'GlobalTimerEvents', 0, 'OffsetValue', 0);
        % PMT shutter timers
        sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', PMT5VDur, 'OnsetDelay', PMTOnsetDelay,...
            'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
            'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMT0VDur,...
            'GlobalTimerEvents', 0, 'OffsetValue', 0);                 
    end
end


    end
end