% MATLAB Script for Auditory Clicks
% Parameters
fs = 100000;  % Sampling frequency (100 kHz)
duration_ms = 10; % Duration of click (milliseconds)
t = 0:1/fs:(duration_ms/1000); % Time vector for the click
amplitude = 1;  % Amplitude of the click

click_idx = zeros(1, length(t));
click_length = 1;
click_train = ones(1, click_length);
click_array = [];
click_array = [click_train, click_idx((click_length + 1):end)];

% Broadband Click
% broadband_click = amplitude * (t == min(t));  % Single impulse
broadband_click = amplitude * (click_array);  % Single impulse


% Tone-Modulated Click
freq = 15000;  % Frequency for tone (15 kHz)
tone_modulated_click = amplitude * sin(2 * pi * freq * t) .* exp(-t*5000); % Gaussian envelope

% Visualization
figure;

% Plot Broadband Click
subplot(2, 2, 1);
plot(t*1000, broadband_click, 'k');
title('Broadband Click (Time Domain)');
xlabel('Time (ms)');
ylabel('Amplitude');


% Plot Tone-Modulated Click
subplot(2, 2, 2);
plot(t*1000, tone_modulated_click, 'b');
title('Tone-Modulated Click (Time Domain)');
xlabel('Time (ms)');
ylabel('Amplitude');


% Spectrogram of Broadband Click
subplot(2, 2, 3);
spectrogram(broadband_click, 128, 120, 256, fs, 'yaxis');
title('Broadband Click (Spectrogram)');


% Spectrogram of Tone-Modulated Click
subplot(2, 2, 4);
spectrogram(tone_modulated_click, 128, 120, 256, fs, 'yaxis');
title('Tone-Modulated Click (Spectrogram)');

% Play the sound through speakers
sound(broadband_click, fs);
sound(broadband_click, fs);
sound(broadband_click, fs);
sound(broadband_click, fs);
% sound(tone_modulated_click, fs);
