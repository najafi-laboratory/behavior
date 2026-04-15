function [wheelPos, wheelSpd, wheelPosShort, wheelSpdShort, wheelPosLong, wheelSpdLong] = ...
    get_wheel_aligned_1(SessionData, tLED, tvec, wheelFieldPref, idxShort, idxLong)
% get_wheel_aligned
% Aligns encoder position to LED onset and returns position/speed matrices.
%
% Outputs:
%   wheelPos      : (nTrials x T) aligned position for all trials
%   wheelSpd      : (nTrials x T) aligned speed for all trials
%   wheelPosShort : (nShort x T) aligned position for short trials
%   wheelSpdShort : (nShort x T) aligned speed for short trials
%   wheelPosLong  : (nLong x T) aligned position for long trials
%   wheelSpdLong  : (nLong x T) aligned speed for long trials

    nTrials = numel(tLED);
    T = numel(tvec);

    wheelPos = nan(nTrials, T);
    wheelSpd = nan(nTrials, T);

    wheelPosShort = nan(0, T);
    wheelSpdShort = nan(0, T);
    wheelPosLong  = nan(0, T);
    wheelSpdLong  = nan(0, T);

    if ~isfield(SessionData,'EncoderData') || isempty(SessionData.EncoderData) || ~iscell(SessionData.EncoderData)
        warning('SessionData.EncoderData not found or not a cell array.');
        return;
    end

    for tr = 1:nTrials
        if ~isfinite(tLED(tr)), continue; end

        % --- get encoder struct for this trial
        enc = get_encoder_struct(SessionData.EncoderData, tr);
        if isempty(enc) || ~isfield(enc,'Times') || isempty(enc.Times)
            continue;
        end

        % --- aligned time (LED at 0)
        t = double(enc.Times(:)) - tLED(tr);

        % --- choose position field
        y = [];
        for k = 1:numel(wheelFieldPref)
            fn = wheelFieldPref{k};
            if isfield(enc, fn) && ~isempty(enc.(fn))
                y = double(enc.(fn)(:));
                break;
            end
        end
        if isempty(y), continue; end

        % --- sort by time
        [t, sidx] = sort(t);
        y = y(sidx);

        % --- interpolate position onto common grid
        wheelPos(tr,:) = interp1(t, y, tvec, 'linear', NaN);

        % --- compute speed
        wheelSpd(tr,:) = gradient(wheelPos(tr,:), tvec);
    end

    % ---- Separate short/long using the provided index arrays
    if nargin >= 5 && ~isempty(idxShort)
        idxShort = idxShort(:);
        idxShort = idxShort(idxShort >= 1 & idxShort <= nTrials);
        wheelPosShort = wheelPos(idxShort,:);
        wheelSpdShort = wheelSpd(idxShort,:);
    end

    if nargin >= 6 && ~isempty(idxLong)
        idxLong = idxLong(:);
        idxLong = idxLong(idxLong >= 1 & idxLong <= nTrials);
        wheelPosLong = wheelPos(idxLong,:);
        wheelSpdLong = wheelSpd(idxLong,:);
    end
end