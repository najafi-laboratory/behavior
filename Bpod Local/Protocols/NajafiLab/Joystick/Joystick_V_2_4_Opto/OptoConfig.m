classdef OptoConfig
   properties
      EnableOpto = 0;
      ComputerHostName = '';
      Rig = '';
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

        function obj = set.ComputerHostName(obj, HostName)
            obj.ComputerHostName = HostName;
            switch obj.ComputerHostName
                case 'COS-3A11406'
                    obj.Rig = 'ImagingRig';
                case 'COS-3A11427'
                    obj.Rig = 'JoystickRig';
            end            
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

        function [AudStimOpto] = GetAudStimOpto(obj, S, OptoTrialType, Vis1Vis2)
            if S.GUI.AudioStimEnable
                AudStimOpto = {'HiFi1', ['P', 6]};
            else
                AudStimOpto = {};
            end
            if obj.EnableOpto
                if OptoTrialType == 2
                    switch Vis1Vis2
                        % vis1 and wait 1 segment stim output
                        case 1
                            if ~S.GUI.OptoVis1 && S.GUI.OptoWaitForPress1
                                AudStimOpto = [AudStimOpto , {'GlobalTimerTrig', '000010001'}];
                            end                             
                        % vis2 and wait 2 segment stim output
                        case 2
                            if ~S.GUI.OptoVis2 && S.GUI.OptoWaitForPress2
                                AudStimOpto = [AudStimOpto , {'GlobalTimerTrig', '001000100'}];
                            end                             
                    end
                end
            end
        end

        function [sma] = InsertGlobalTimer(obj, sma, S, VisStim)
            if obj.EnableOpto 
                % shutter close delays
                ShutterPulseWidthAdd = 0.003; % add 3ms to PMTCloseDelay for shutter to re-open
                PMTStartCloseDelay = 0.010;
                PMTCloseTransferDelay = 0.0121;
                % PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay + ShutterPulseWidthAdd;
                PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;
                PMTMin5VSignalDur = PMTCloseDelay + 0.003; % ~25.1ms, minimum measured duration of 5V shutter signal for shutter to re-open 

                % shutter open delays
                PMTStartOpenDelay = 0.0078;
                PMTOpenTransferDelay = 0.0125;
                
                % scope 3-frame segment dur
                ScopeFrameDuration = 0.033;

                % initial gray frame vis stim offset, statistical delay of
                % 2 frames at 60fps
                %VisStimShift = 0.0147 + 0.0353098; % f1 and f2,f3
                % VisStimShift = 0.0363000; % f2,f3
                % VisStimShift = 0.031698; % f1,f2 - joystick rig
                % VisStimShift = 0.0329645; % f1,f2 - imaging rig
                % % VisStimDurationOffset = 0.002; % ~1.5ms measured vis stim offset
                % VisStimDurationOffset = 0.0014; % ~1.5ms measured vis stim offset 

                % initial gray frame vis stim offset, statistical delay of
                % 2 frames at 60fps
                switch obj.Rig
                    case 'ImagingRig'
                        VisStimShift = 0.0329645; % f1 + f2 - imaging rig
                        VisStimDurationOffset = 0.0014; % ~measured vis stim offset from 100ms
                    case 'JoystickRig'
                        VisStimShift = 0.031698; % f1 + f2 - joystick rig
                        VisStimDurationOffset = 0.0014; % ~measured vis stim offset from 100ms
                end

                % LED1OnsetDelay = PMTCloseDelay;
                % LED2OnsetDelay = PMTCloseDelay;

                LED1OnsetDelay = 0;
                LED2OnsetDelay = 0;                

                PMT1OnsetDelay = 0;
                PMT2OnsetDelay = 0;

                % bpod timer loop limit
                MaxLoopNum = 255;
                MinLoopIncrement = S.GUI.MaxOptoDur/MaxLoopNum;

                if S.GUI.OptoVis1
                    % LED1OnsetDelay = LED1OnsetDelay + VisStimShift - PMTCloseDelay;
                    LED1OnsetDelay = VisStimShift;
                    % PMT1OnsetDelay = PMT1OnsetDelay + VisStimShift - PMTCloseDelay;
                    PMT1OnsetDelay = LED1OnsetDelay - PMTCloseDelay;
                end

                if ~S.GUI.OptoVis1 && S.GUI.OptoWaitForPress1
                    LED1OnsetDelay = VisStim.VisStimDuration - VisStimDurationOffset;
                    % PMT1OnsetDelay = VisStim.VisStimDuration - PMTCloseDelay - VisStimDurationOffset;
                    PMT1OnsetDelay = LED1OnsetDelay - PMTCloseDelay;
                end

                if S.GUI.OptoVis2
                    % LED2OnsetDelay = LED2OnsetDelay + VisStimShift - PMTCloseDelay;
                    LED2OnsetDelay = VisStimShift;
                    % PMT2OnsetDelay = PMT2OnsetDelay + VisStimShift - PMTCloseDelay;
                    PMT2OnsetDelay = LED2OnsetDelay - PMTCloseDelay;
                end

                if ~S.GUI.OptoVis2 && S.GUI.OptoWaitForPress2
                    LED2OnsetDelay = VisStim.VisStimDuration - VisStimDurationOffset;
                    % PMT2OnsetDelay = VisStim.VisStimDuration - PMTCloseDelay - VisStimDurationOffset;
                    PMT2OnsetDelay = LED2OnsetDelay - PMTCloseDelay;
                end

                switch S.GUI.PulseType
                    case 1
                        % OnDur = VisStim.VisStimDuration - PMTCloseDelay;
                        LEDOffDur = 0;
                        
                        % LoopLED1 = 1;
                        LoopLED2 = 1;
                        % LoopPMT1 = 1;
                        LoopPMT2 = 1;
                       
                        % PMTCloseDur = VisStim.VisStimDuration;
                        PMTOffDur = 0;

                        OnDur = MinLoopIncrement;
                        LoopLED1 = floor(S.GUI.MaxOptoDur/OnDur);

                        PMTCloseDur = MinLoopIncrement;
                        LoopPMT1 = floor((S.GUI.MaxOptoDur+PMTCloseDur)/PMTCloseDur);

                        LoopLED1 = 0;
                        LoopLED2 = 0;
                        LoopPMT1 = 0;
                        LoopPMT2 = 0;

                        OnDur = round(S.GUI.MaxOptoDur, 4);
                        LEDOffDur = 0;

                        PMTOffDur = 0;
                        PMTCloseDur = round(S.GUI.MaxOptoDur + PMTCloseDelay - PMTStartOpenDelay, 4);

                    case 2
                        T = 1/S.GUI.CS_SquareFreq_Hz;
                        OnDur = S.GUI.CS_SquareOnDur_ms/1000;
                        LEDOffDur = abs(OnDur - T);

                        % add fix to calculation for square wave shutter close dur 

                        % S.GUI.MaxOptoDur = 0.05;
                        % get integer number of loops for LED cycles to be
                        % within max opto dur
                        LoopLED1 = floor((S.GUI.MaxOptoDur/(OnDur+LEDOffDur)));
                        LoopLED2 = LoopLED1;
                        LoopPMT1 = 0;
                        LoopPMT2 = 0;
                       
                        TimeFromLEDOffset = S.GUI.MaxOptoDur - ((T * LoopLED1) - LEDOffDur);

                        PMTCloseDur = round(S.GUI.MaxOptoDur + PMTCloseDelay - PMTStartOpenDelay - TimeFromLEDOffset, 4);
                        PMTOffDur = 0;                        
                    case 3
                        OnDur = S.GUI.PS_LEDOnPulseDur;
                        % LEDOffDur = PMTOpenTransferDelay + 2*ScopeFrameDuration + PMTCloseTransferDelay;
                        LEDOffDur = S.GUI.PS_SquareFreq_s - OnDur;
                        
                        if ~S.GUI.OptoWaitForPress1
                            LoopLED1 = 0;
                            LoopPMT1 = 0;
                        else
                            LoopLED1 = 1;
                            LoopPMT1 = 1;
                        end

                        if ~S.GUI.OptoWaitForPress2
                            LoopLED2 = 0;
                            LoopPMT2 = 0;
                        else
                            LoopLED2 = 1;
                            LoopPMT2 = 1;
                        end
                       
                        PMTCloseDur = PMTCloseDelay;
                        % if the LED is on for longer than the shutter StartOpenDelay,
                        % then increase shutter 5V duration by the difference (PS_LEDOnPulseDur - PMTStartOpenDelay)
                        if OnDur > PMTStartOpenDelay
                            PMTCloseDur = PMTCloseDur + (OnDur - PMTStartOpenDelay);
                        end
                         % if shutter duration is less than the minimum dur for the
                        % shutter to re-open, set it to minimum shutter pulse dur
                        PMTCloseDur = max(PMTCloseDur, PMTMin5VSignalDur);

                        % PMTOffDur = OnDur + PMTOpenTransferDelay + 2*ScopeFrameDuration - PMTStartCloseDelay - ShutterPulseWidthAdd;
                        % PMTOffDur = OnDur + PMTOpenTransferDelay + 2*ScopeFrameDuration - PMTStartCloseDelay - ShutterPulseWidthAdd;

                        % duration of LED pulse 0V is cycle period minus 5V dur
                        PMTOffDur =  S.GUI.PS_SquareFreq_s - PMTCloseDur;
                end 

                % LED timers
                % seg 1
                sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', OnDur, 'OnsetDelay', LED1OnsetDelay,...
                    'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                    'Loop', LoopLED1, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
                    'GlobalTimerEvents', 0, 'OffsetValue', 0);

                % seg 2
                sma = SetGlobalTimer(sma, 'TimerID', 3, 'Duration', OnDur, 'OnsetDelay', LED2OnsetDelay,...
                    'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                    'Loop', LoopLED2, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
                    'GlobalTimerEvents', 0, 'OffsetValue', 0);

                % shutter timers
                % seg 1
                sma = SetGlobalTimer(sma, 'TimerID', 5, 'Duration', PMTCloseDur, 'OnsetDelay', PMT1OnsetDelay,...
                    'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                    'Loop', LoopPMT1, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMTOffDur,...
                    'GlobalTimerEvents', 0, 'OffsetValue', 0);     

                % seg 2
                sma = SetGlobalTimer(sma, 'TimerID', 7, 'Duration', PMTCloseDur, 'OnsetDelay', PMT2OnsetDelay,...
                    'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                    'Loop', LoopPMT2, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMTOffDur,...
                    'GlobalTimerEvents', 0, 'OffsetValue', 0); 

                % shutter reset timer
                sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', 0.030, 'OnsetDelay', 0,...
                    'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                    'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
                    'GlobalTimerEvents', 0, 'OffsetValue', 0);                 
            end
        end
    end
end
