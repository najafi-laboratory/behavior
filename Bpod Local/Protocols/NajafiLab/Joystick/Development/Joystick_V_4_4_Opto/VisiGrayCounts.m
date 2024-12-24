

VisDetectGray1Dur = [];
VisDetectGray2Dur = [];

VisStimInterruptCount = 0;

for trial = 1:SessionData.nTrials

    VisInterrupt = SessionData.RawEvents.Trial{1, trial}.States.VisStimInterrupt;
    if ~isnan(VisInterrupt)
        VisStimInterruptCount = VisStimInterruptCount + 1;
    end

    VisDetectGray1 = SessionData.RawEvents.Trial{1, trial}.States.VisDetectGray1;
    if ~isnan(VisDetectGray1)
        VisDetectGray1Dur = [VisDetectGray1Dur (VisDetectGray1(2) - VisDetectGray1(1))];
    end

    VisDetectGray2 = SessionData.RawEvents.Trial{1, trial}.States.VisDetectGray2;    
    if ~isnan(VisDetectGray2)
       VisDetectGray2Dur = [VisDetectGray2Dur (VisDetectGray2(2) - VisDetectGray2(1))];
    end



end

VisStimInterruptCount

max(VisDetectGray1Dur)
min(VisDetectGray1Dur)
mean(VisDetectGray1Dur)
std(VisDetectGray1Dur)
var(VisDetectGray1Dur)

figure(1)
plot(VisDetectGray1Dur)

max(VisDetectGray2)
min(VisDetectGray2)
mean(VisDetectGray2)
std(VisDetectGray2)
var(VisDetectGray2)

figure(2)
plot(VisDetectGray2)