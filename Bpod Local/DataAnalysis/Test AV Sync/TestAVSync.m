


F1_samples = [];
F2_samples = [];
Vis1AsMeasuredByBNC_samples = [];
Stim_BNC_Diff_samples = [];

for trial = 1:SessionData.nTrials
    
    % trial = 5; % test individual trial

    Events = SessionData.RawEvents.Trial{1, trial}.Events;
    States = SessionData.RawEvents.Trial{1, trial}.States;
    
    % BNC1High for trial
    BNC1High = Events.BNC1High;
    % BNC1Low for trial
    BNC1Low = Events.BNC1Low;

    % First gray frame
    F1 = BNC1Low(1) - BNC1High(1);
    % Second gray frame
    F2 = BNC1High(2) - BNC1Low(1);
    % BNC measured duration of gray frames
    GrayShift = F1 + F2;
    % BNC measured duration of grating frames
    Grating = BNC1Low(2) - BNC1High(2);
    % BNC measured duration of Vis1
    Vis1AsMeasuredByBNC = GrayShift + Grating;
        
    disp(['F1 ' num2str(F1)]);
    disp(['F2 ' num2str(F2)]);
    disp(['GrayShift ' num2str(GrayShift)]);
    disp(['Grating ' num2str(Grating)]);
    disp(['Vis1AsMeasuredByBNC ' num2str(Vis1AsMeasuredByBNC)]);
    
    VisualStimulus1 = States.VisualStimulus1;
    % State timer measured duration of VisualStimulus1 (state duration)
    Stim1Dur = VisualStimulus1(2) - VisualStimulus1(1);
    disp(['Stim1Dur ' num2str(Stim1Dur)]);
    
    % Difference between state timer and BNC measure of visual stimulus
    Stim_BNC_Diff = Stim1Dur - Vis1AsMeasuredByBNC;    
    disp(['Stim_BNC_Diff ' num2str(Stim_BNC_Diff)]);
    
    % Difference between photodiode detection of first frame and start of
    % visual stimulus state
    BNC_vis_start_diff = BNC1High(1) - VisualStimulus1(1);
    % Difference between photodiode detection of background gray frame and
    % end of visual stimulus state
    BNC_vis_stop_diff = BNC1Low(2) - VisualStimulus1(2);
    disp(['BNC_vis_start_diff ' num2str(BNC_vis_start_diff)]);
    disp(['BNC_vis_stop_diff ' num2str(BNC_vis_stop_diff)]);
        
    VisualStimulus2 = States.VisualStimulus2;

    % stat samples
    F1_samples = [F1_samples F1];
    F2_samples = [F2_samples F2];
    Vis1AsMeasuredByBNC_samples = [Vis1AsMeasuredByBNC_samples Vis1AsMeasuredByBNC];
    Stim_BNC_Diff_samples= [Stim_BNC_Diff_samples Stim_BNC_Diff];
end

disp(['max(F1_samples) ' num2str(max(abs(F1_samples)))]);
disp(['min(F1_samples) ' num2str(min(abs(F1_samples)))]);
disp(['mean(F1_samples) ' num2str(mean(abs(F1_samples)))]);
disp(['std(F1_samples) ' num2str(std(abs(F1_samples)))]);
disp(['var(F1_samples) ' num2str(var(abs(F1_samples)))]); 

disp(['max(F2_samples) ' num2str(max(abs(F2_samples)))]);
disp(['min(F2_samples) ' num2str(min(abs(F2_samples)))]);
disp(['mean(F2_samples) ' num2str(mean(abs(F2_samples)))]);
disp(['std(F2_samples) ' num2str(std(abs(F2_samples)))]);
disp(['var(F2_samples) ' num2str(var(abs(F2_samples)))]);  

disp(['max(Vis1AsMeasuredByBNC_samples) ' num2str(max(abs(Vis1AsMeasuredByBNC_samples)))]);
disp(['min(Vis1AsMeasuredByBNC_samples) ' num2str(min(abs(Vis1AsMeasuredByBNC_samples)))]);
disp(['mean(Vis1AsMeasuredByBNC_samples) ' num2str(mean(abs(Vis1AsMeasuredByBNC_samples)))]);
disp(['std(Vis1AsMeasuredByBNC_samples) ' num2str(std(abs(Vis1AsMeasuredByBNC_samples)))]);
disp(['var(Vis1AsMeasuredByBNC_samples) ' num2str(var(abs(Vis1AsMeasuredByBNC_samples)))]);   

disp(['max(Stim_BNC_Diff_samples) ' num2str(max(abs(Stim_BNC_Diff_samples)))]);
disp(['min(Stim_BNC_Diff_samples) ' num2str(min(abs(Stim_BNC_Diff_samples)))]);
disp(['mean(Stim_BNC_Diff_samples) ' num2str(mean(abs(Stim_BNC_Diff_samples)))]);
disp(['std(Stim_BNC_Diff_samples) ' num2str(std(abs(Stim_BNC_Diff_samples)))]);
disp(['var(Stim_BNC_Diff_samples) ' num2str(var(abs(Stim_BNC_Diff_samples)))]);    
    
figure('Name', 'F1_samples', 'NumberTitle', 'off')
histogram(F1_samples)
figure('Name', 'F2_samples', 'NumberTitle', 'off')
histogram(F2_samples)
figure('Name', 'Vis1AsMeasuredByBNC_samples', 'NumberTitle', 'off')
histogram(Vis1AsMeasuredByBNC_samples)
figure('Name', 'Stim_BNC_Diff_samples', 'NumberTitle', 'off')
histogram(Stim_BNC_Diff_samples)
