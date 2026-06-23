
S.GUI.ISIShortMin_s = 0.500;
S.GUI.ISIShortMax_s = 1.250; 

% Generate 5 evenly spaced values including min and max
values = linspace(S.GUI.ISIShortMin_s, S.GUI.ISIShortMax_s, 5);

% Randomly select one of the values
selected_value = values(randi(5));

% Display the result
disp('Selected value:');
disp(selected_value);
% % 
% % 
% % if 0 
% %     lb = 0
% %     ub = 1
% % 
% %     a = lb;  % Lower bound
% %     b = ub;  % Upper bound    
% %     r = a + (b-a)*rand();  % Single random value between a and b  
% %     round(r)
% % end
% % 
% % 
% % nTrials = SessionData.nTrials;
% 
% 
% % H = BpodHiFi(BpodSystem.ModuleUSB.HiFi1);
% 
% SF = 44100;
% % SF = 192000;
% 
% duration_ms = 1; % Duration of click (milliseconds)
% t = 0:1/SF:(duration_ms/1000); % Time vector for the click
% amplitude = 1;  % Amplitude of the click
% % Tone-Modulated Click
% freq = 16000;  % Frequency for tone (16 kHz)
% ClickAudio = [];
% ClickAudio = amplitude * sin(2 * pi * freq * t) .* exp(-t*5000); % Gaussian envelope
% 
% 
% 
% 
% click_zeros = zeros(1, 300);
% % click_zeros = [];
% num_clicks = 10;
% for i = 1:num_clicks
%     % ClickAudio = [ClickAudio click_zeros ClickAudio];
%     ClickAudio = [ClickAudio 0 0 0 ClickAudio];
%     % ClickAudio = [ClickAudio click_zeros];
% end
% 
% H.load(6, ClickAudio);
% H.push();
% 
% H.play(6);
% H.play(6);
% H.play(6);
% H.play(6);
% 
% % sound(ClickAudio, SF);
% 
% if 0 
%     ClickAudio = [];
%     click_zeros = zeros(1, 1000);
%     broadband_click = amplitude * (t == min(t));  % Single impulse
%     ClickAudio = broadband_click;
%     num_clicks = 5;
%     for i = 1:num_clicks
%         % ClickAudio = [ClickAudio click_zeros ClickAudio];
%         ClickAudio = [ClickAudio click_zeros ClickAudio];
%         % ClickAudio = [ClickAudio click_zeros];
%     end
% 
%     % H = []
%     H.load(6, ClickAudio);
% 
%     H.play(6);
%     H.play(6);
%     H.play(6);
%     H.play(6);
% end
% 
% 
% 
% fs = 44100;  % Sampling frequency (44.1 kHz for high-quality audio)
% clickFrequency = 16000;  % Frequency of the click sound in Hz
% clickDuration = 0.55;   % Duration of the click sound in seconds
% clickAmplitude = 0.5;   % Amplitude of the click sound (volume)
% 
% % Create the clicking sound signal
% t = 0:1/fs:clickDuration;  % Time vector for the click duration
% clickSignal = clickAmplitude * sin(2 * pi * clickFrequency * t); 
% 
% clickSignal = [clickSignal clickSignal clickSignal clickSignal clickSignal clickSignal clickSignal];
% 
% 
% % sound(clickSignal, SF);
% 
% 
% 
% 
% 
% 
% 
% H.load(7, clickSignal);
% H.push();
% H.play(7);