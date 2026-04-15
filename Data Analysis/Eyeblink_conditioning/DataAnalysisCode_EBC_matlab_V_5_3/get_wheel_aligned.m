function [wheelPos, wheelSpd] = get_wheel_aligned(SessionData, tLED, tvec, wheelFieldPref)
% Returns aligned wheel position and speed matrices: (nTrials x numel(tvec))
    nTrials = numel(tLED);
    wheelPos = nan(nTrials, numel(tvec));
    wheelSpd = nan(nTrials, numel(tvec));

    if ~isfield(SessionData,'EncoderData') || isempty(SessionData.EncoderData)
        warning('SessionData.EncoderData not found.');
        return;
    end

    for tr = 1:nTrials
        if isnan(tLED(tr)), continue; end

        % EncoderData sometimes stored as cell {trial,device} or {device,trial}.
        enc = get_encoder_struct(SessionData.EncoderData, tr);

        if isempty(enc) || ~isfield(enc,'Times')
            continue;
        end

        t = double(enc.Times(:)) - tLED(tr);  % align to LED
        y = [];

        % pick best available position field
        for k = 1:numel(wheelFieldPref)
            fn = wheelFieldPref{k};
            if isfield(enc, fn) && ~isempty(enc.(fn))
                y = double(enc.(fn)(:));
                break;
            end
        end
        if isempty(y), continue; end

        % Sort in case
        [t, idx] = sort(t);
        y = y(idx);

        % Interpolate position onto common tvec
        wheelPos(tr,:) = interp1(t, y, tvec, 'linear', NaN);

        % Speed = derivative of position
        % Use gradient on interpolated trace for stability
        wheelSpd(tr,:) = gradient(wheelPos(tr,:), tvec);
    end
end