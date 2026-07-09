function isProbe = decide_probe_trial_simple(S, blocks, TrialTypeSequence, currentTrial, currentBlockIndex, currentTrialInBlock, numWarmupTrials)
% True if the CURRENT TRIAL should be a probe.
% - Single-block (TS==1): use trials since warmup (session may stop early).
% - Multi/double-block (TS~=1): use block-relative indexing (NO warmup offset).
% Rules:
%   - Skip first lead-in trials per block
%   - Never probe on the LAST trial of a block (for multi-block)
%   - Keep ~p% probes at any stop time (online quota via ceil)
%   - Enforce >= ProbeMinSeparation non-probes between probes (block-relative)
    isProbe = false;
    if ~isfield(S,'GUI') || S.GUI.UseProbeTrials ~= 1, return; end
    p    = S.GUI.probetrials_percentage_perBlock / 100;  % e.g., 0.15
    lead = S.GUI.num_initial_nonprobe_trials_perBlock;   % skip first N per block
    sep  = S.GUI.ProbeMinSeparation + 1;                 % index separation
    if isempty(numWarmupTrials), numWarmupTrials = 0; end
    % Persistent per-block (or 1 slot for single-block) state
    persistent seen_byBlock taken_byBlock lastRel_byBlock init_len init_seq
    if isempty(init_len) || isempty(init_seq) || init_len ~= numel(blocks) || init_seq ~= TrialTypeSequence
        seen_byBlock    = zeros(1, max(1,numel(blocks)));
        taken_byBlock   = zeros(1, max(1,numel(blocks)));
        lastRel_byBlock = -inf(1, max(1,numel(blocks)));
        init_len = numel(blocks);
        init_seq = TrialTypeSequence;
    end
    if TrialTypeSequence == 1
        % ---------- SINGLE-BLOCK: trials since warmup ----------
        bIdx = 1;                                   % single slot
        rel  = currentTrial - numWarmupTrials;      % 1,2,3,... after warmup
        isCand = (rel >= 1 + lead);                 % no need to forbid "last"
        if rel == 1                                  % reset at single-block start
            seen_byBlock(bIdx) = 0; taken_byBlock(bIdx) = 0; lastRel_byBlock(bIdx) = -inf;
        end
    else
        % ---------- MULTI/DOUBLE-BLOCK: strict block-relative ----------
        bIdx = max(1, currentBlockIndex);
        rel  = currentTrialInBlock;                 % 1..blocks(bIdx)
        blockLen = blocks(bIdx);
        isCand = (rel >= 1 + lead) && (rel < blockLen); % forbid last trial in block
        if rel == 1                                   % reset at each new block start
            seen_byBlock(bIdx) = 0; taken_byBlock(bIdx) = 0; lastRel_byBlock(bIdx) = -inf;
        end
    end
    if ~isCand, return; end
    seen  = seen_byBlock(bIdx) + 1;            % we are considering this candidate now
    taken = taken_byBlock(bIdx);
    lastR = lastRel_byBlock(bIdx);
    quota = ceil(p * seen);                    % desired probes among seen candidates
    need  = quota - taken;
    canPlace = ( (rel - lastR) >= sep );       % spacing in block-relative indices
    if (need > 0) && canPlace
        isProbe = true;
        taken   = taken + 1;
        lastR   = rel;
    end
    % write back
    seen_byBlock(bIdx)    = seen;
    taken_byBlock(bIdx)   = taken;
    lastRel_byBlock(bIdx) = lastR;
end