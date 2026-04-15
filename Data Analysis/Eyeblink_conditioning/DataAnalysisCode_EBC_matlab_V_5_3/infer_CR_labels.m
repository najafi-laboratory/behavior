function CR = infer_CR_labels(SessionData, nTrials)
% Tries to find CR outcome if stored. Otherwise returns false.
    CR = false(nTrials,1);

    % Example: SessionData.Custom.TrialOutcome or similar
    if isfield(SessionData,'Custom')
        C = SessionData.Custom;
        if isfield(C,'CR_isGood') && numel(C.CR_isGood) >= nTrials
            CR = logical(C.CR_isGood(:));
            return;
        end
        if isfield(C,'GoodCR') && numel(C.GoodCR) >= nTrials
            CR = logical(C.GoodCR(:));
            return;
        end
    end
end