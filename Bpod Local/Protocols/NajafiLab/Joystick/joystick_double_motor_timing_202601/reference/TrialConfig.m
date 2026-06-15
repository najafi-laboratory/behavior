classdef TrialConfig
    methods
        function TrialTypes = GenTrials(~, S, MaxTrials, numTrialTypes, TrialTypes, currentTrial, updateTrialTypeSequence)
            if S.GUI.EnableManualTrialType
                TrialTypes(currentTrial:end) = S.GUI.ManualTrialType;
                return
            end

            if ~updateTrialTypeSequence
                return
            end

            nRemaining = MaxTrials - currentTrial + 1;
            switch S.GUI.TrialTypeSequence
                case 1
                    TrialTypes(currentTrial:end) = randi(numTrialTypes, 1, nRemaining);
                case 2
                    firstType = randi(numTrialTypes);
                    TrialTypes(currentTrial:end) = makeAlternatingBlocks(S, MaxTrials, nRemaining, firstType, 3 - firstType);
                case 3
                    TrialTypes(currentTrial:end) = makeAlternatingBlocks(S, MaxTrials, nRemaining, 1, 2);
                case 4
                    TrialTypes(currentTrial:end) = makeAlternatingBlocks(S, MaxTrials, nRemaining, 2, 1);
            end
        end

        function ProbeTrialTypes = GenProbeTrials(~, S, TrialTypes, currentTrial)
            ProbeTrialTypes = zeros(1, S.GUI.MaxTrials);
            if S.GUI.ProbeTrialFraction <= 0
                return
            end

            sequence = zeros(1, S.GUI.MaxTrials);
            writeIdx = 1;
            edgeOffset = 2;
            blockStarts = [1, find(abs(diff(TrialTypes)) == 1) + 1, S.GUI.MaxTrials + 1];
            blockLengths = diff(blockStarts);

            for blockIdx = 1:numel(blockLengths)
                blockLength = blockLengths(blockIdx);
                blockProbeFlags = zeros(1, blockLength);
                eligible = edgeOffset + 1:blockLength - edgeOffset;
                nProbeTrials = ceil(S.GUI.ProbeTrialFraction * numel(eligible));
                if ~isempty(eligible) && nProbeTrials > 0
                    blockProbeFlags(eligible(randperm(numel(eligible), nProbeTrials))) = 1;
                end

                stopIdx = min(S.GUI.MaxTrials, writeIdx + blockLength - 1);
                sequence(writeIdx:stopIdx) = blockProbeFlags(1:(stopIdx - writeIdx + 1));
                writeIdx = stopIdx + 1;
                if writeIdx > S.GUI.MaxTrials
                    break
                end
            end

            ProbeTrialTypes(currentTrial:end) = sequence(currentTrial:end);
        end

        function ITI = GetITI(~, S)
            if S.GUI.SetManualITI == 1
                if isnumeric(S.GUI.ManualITI)
                    ITI = S.GUI.ManualITI;
                else
                    ITI = str2double(S.GUI.ManualITI);
                end
                if isnan(ITI)
                    ITI = 0;
                end
                return
            end

            ITI = inf;
            while ITI < S.GUI.ITIMin || ITI > S.GUI.ITIMax
                ITI = -log(rand) * S.GUI.ITIMean;
            end
        end
    end
end

function sequence = makeAlternatingBlocks(S, MaxTrials, nRemaining, firstType, secondType)
    sequence = zeros(1, MaxTrials);
    writeIdx = 1;
    blockIdx = 1;
    isFirstBlock = S.GUI.NumEasyWarmupTrials > 0;
    blockJitter = randi([-S.GUI.BlockLengthMargin, S.GUI.BlockLengthMargin], 1, MaxTrials);
    blockLengths = S.GUI.NumTrialsPerBlock + blockJitter;

    while writeIdx <= MaxTrials
        firstLength = blockLengths(blockIdx) + isFirstBlock * S.GUI.NumEasyWarmupTrials;
        secondLength = blockLengths(blockIdx + 1);
        block = [repmat(firstType, 1, firstLength), repmat(secondType, 1, secondLength)];
        stopIdx = min(MaxTrials, writeIdx + numel(block) - 1);
        sequence(writeIdx:stopIdx) = block(1:(stopIdx - writeIdx + 1));
        writeIdx = stopIdx + 1;
        blockIdx = blockIdx + 2;
        isFirstBlock = false;
    end

    sequence = sequence(1:nRemaining);
end
