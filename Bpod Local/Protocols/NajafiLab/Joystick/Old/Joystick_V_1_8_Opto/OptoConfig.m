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

        function [AudStimOpto] = GetAudStimOpto(obj, S, OptoTrialType, Vis1Vis2)
            if S.GUI.AudioStimEnable
                AudStimOpto = {'HiFi1', ['P', 6]};
            else
                AudStimOpto = {};
            end
            if obj.EnableOpto
                if OptoTrialType == 2
                    switch Vis1Vis2
                        case 1
                            if S.GUI.OptoVis1
                                % AudStimOpto = [AudStimOpto , {'GlobalTimerTrig', '000010001'}];
                            end
                        case 2
                            if S.GUI.OptoVis2
                                % AudStimOpto = [AudStimOpto , {'GlobalTimerTrig', '001000100'}];
                            end
                    end
                end
            end
        end

        function [sma] = InsertGlobalTimer(obj, sma, S, VisStim)
            if obj.EnableOpto 
                % shutter close delays
                PMTStartCloseDelay = 0.010;
                PMTCloseTransferDelay = 0.0121;
                PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;

                % shutter open delays
                PMTStartOpenDelay = 0.0078;
                PMTOpenTransferDelay = 0.0125;
                
                % scope 3-frame segment dur
                ScopeFrameDuration = 0.033;

                % initial gray frame vis stim offset, statistical delay of
                % 2 frames at 60fps
                VisStimShift = 0.0147 + 0.0353098; % f1 and f2,f3

                LED1OnsetDelay = PMTCloseDelay;
                LED2OnsetDelay = PMTCloseDelay;

                PMT1OnsetDelay = 0;
                PMT2OnsetDelay = 0;

                if S.GUI.OptoVis1
                    LED1OnsetDelay = LED1OnsetDelay + VisStimShift - PMTCloseDelay;
                    PMT1OnsetDelay = PMT1OnsetDelay + VisStimShift - PMTCloseDelay;
                end

                if S.GUI.OptoVis2
                    LED2OnsetDelay = LED2OnsetDelay + VisStimShift - PMTCloseDelay;
                    PMT2OnsetDelay = PMT2OnsetDelay + VisStimShift - PMTCloseDelay;
                end

                switch S.GUI.PulseType
                    case 1
                        OnDur = VisStim.VisStimDuration - PMTCloseDelay;
                        LEDOffDur = 0;
                        
                        LoopLED1 = 1;
                        LoopLED2 = 1;
                        LoopPMT1 = 1;
                        LoopPMT2 = 1;
                       
                        PMTCloseDur = VisStim.VisStimDuration;
                        PMTOffDur = 0;
                    case 2
                        T = 1/S.GUI.PulseFreq_Hz;
                        OnDur = S.GUI.PulseOnDur_ms/1000;
                        LEDOffDur = abs(OnDur - T);

                        LoopLED1 = 1;
                        LoopLED2 = 1;
                        LoopPMT1 = 1;
                        LoopPMT2 = 1;

                        PMTCloseDur = VisStim.VisStimDuration;
                        PMTOffDur = 0;                        
                    case 3
                        OnDur = PMTStartOpenDelay;
                        LEDOffDur = PMTOpenTransferDelay + 2*ScopeFrameDuration + PMTCloseTransferDelay;
                        
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
                        PMTOffDur = OnDur + PMTOpenTransferDelay + 2*ScopeFrameDuration - PMTStartCloseDelay;
                end 

                % onset delay for vis1/2 35.3098
              
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
            end
        end
    end
end
