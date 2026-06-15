function EncoderData = CompleteEncoderData(EncoderData, TrialDuration, PreviousEncoderData)
if ~isempty(EncoderData.Times)
    if EncoderData.Times(1) > 0
        if isempty(PreviousEncoderData)
            startPosition = EncoderData.Positions(1);
        else
            startPosition = PreviousEncoderData.Positions(end);
        end
        EncoderData.Times = [0, EncoderData.Times];
        EncoderData.Positions = [startPosition, EncoderData.Positions];
        EncoderData.nPositions = EncoderData.nPositions + 1;
    end

    if EncoderData.Times(end) < TrialDuration
        EncoderData.Times = [EncoderData.Times, TrialDuration];
        EncoderData.Positions = [EncoderData.Positions, EncoderData.Positions(end)];
        EncoderData.nPositions = EncoderData.nPositions + 1;
    end
    return
end

if isempty(PreviousEncoderData)
    position = 0;
else
    position = PreviousEncoderData.Positions(end);
end

EncoderData.Times = [0, TrialDuration];
EncoderData.Positions = [position, position];
EncoderData.nPositions = 2;
end
