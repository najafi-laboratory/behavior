function fnExcludedTrials(obj, source,event)
    global BpodSystem
    global S
    S = BpodParameterGUI('sync', S);
    BpodSystem.Data.Excluded = S.GUI.numExcludedTrials;
end


