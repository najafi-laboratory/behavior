function enc = get_encoder_struct(EncoderData, tr)
% Handles different cell array shapes; chooses the non-empty encoder entry.
    enc = [];
    if ~iscell(EncoderData)
        return;
    end

    sz = size(EncoderData);

    % Common: EncoderData{trial, device}
    if numel(sz) == 2 && tr <= sz(1)
        row = tr;
        for col = 1:sz(2)
            if ~isempty(EncoderData{row,col}) && isstruct(EncoderData{row,col})
                enc = EncoderData{row,col};
                return;
            end
        end
    end

    % Alternative: EncoderData{device, trial}
    if numel(sz) == 2 && tr <= sz(2)
        col = tr;
        for row = 1:sz(1)
            if ~isempty(EncoderData{row,col}) && isstruct(EncoderData{row,col})
                enc = EncoderData{row,col};
                return;
            end
        end
    end
end