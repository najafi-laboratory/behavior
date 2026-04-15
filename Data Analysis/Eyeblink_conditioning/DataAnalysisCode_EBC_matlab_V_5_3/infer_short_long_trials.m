function [isShort, isLong] = infer_short_long_trials(SessionData, nTrials)
% Best-effort inference. Replace with your own short/long logic if needed.
% Many EBC protocols store short/long in TrialTypes (e.g., 1=short, 2=long).
    isShort = false(nTrials,1);
    isLong  = false(nTrials,1);

    if isfield(SessionData,'TrialTypes') && numel(SessionData.TrialTypes) >= nTrials
        tt = SessionData.TrialTypes(:);
        % Heuristic: smaller value = short, larger = long
        u = unique(tt(~isnan(tt)));
        if numel(u) >= 2
            isShort = tt == min(u);
            isLong  = tt == max(u);
            return;
        end
    end

    % Fallback: everything unknown -> mark all as "short" to avoid empty plots
    isShort(:) = true;
    isLong(:)  = false;
end