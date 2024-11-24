function [CR_plus] = CR_plus_eval(time,signal,t1,t2,t_LED,t_puff,CR_threshold)
CR_plus = false;


% Find indices corresponding to t1 and t2
indices = find(time >= t1 & time <= t2);

% Extract the relevant portion of the signal
signal_subset = signal(indices);

% Find indices corresponding to t1 and t2
indices = find(time >= t_LED & time <= t_puff);

% Extract the relevant portion of the signal
signal_LEDpuff = signal(indices);


% Calculate the average of the signal between t1 and t2
baseline = mean(signal_subset);


% Check if any signal values exceed the baseline
if any(signal_LEDpuff > (baseline+CR_threshold))
    CR_plus = true;
end


end