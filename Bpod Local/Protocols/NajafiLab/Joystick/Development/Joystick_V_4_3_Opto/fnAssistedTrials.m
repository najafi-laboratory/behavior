function fnAssistedTrials(obj, source,event)
    global BpodSystem
    global S
    S = BpodParameterGUI('sync', S);
    BpodSystem.Data.Assisted(S.GUI.ATRangeStart:S.GUI.ATRangeStop) = 1;
end


