function tLED = get_led_onset_time(SessionData, tr, stateCandidates)
% Try to find LED onset time from RawEvents states.
    tLED = NaN;

    if ~isfield(SessionData,'RawEvents') || ~isfield(SessionData.RawEvents,'Trial')
        return;
    end
    T = SessionData.RawEvents.Trial{tr};
    if ~isfield(T,'States') || isempty(T.States)
        return;
    end

    st = T.States;
    for k = 1:numel(stateCandidates)
        nm = stateCandidates{k};
        if isfield(st, nm) && ~isempty(st.(nm)) && size(st.(nm),2) >= 1
            % Bpod state times are usually [tEntry tExit]
            tLED = st.(nm)(1,1);
            return;
        end
    end
end
