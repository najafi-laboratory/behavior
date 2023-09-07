% ITI exponential distribution Parameters
minITI = 1;    % Minimum ITI (in seconds)
maxITI = 5;    % Maximum ITI (in seconds)
lambda = 0.3;  % Lambda parameter of the exponential distribution

% Generate ITIs
numTrials = 10;  % Number of trials
ITIs = zeros(1, numTrials);  % Initialize array to store ITIs

for i = 1:numTrials
    % Generate a random value from the exponential distribution
    ITI = -log(rand) / lambda;
    
    % Check if the generated ITI is within the desired range
    while ITI < minITI || ITI > maxITI
        ITI = -log(rand) / lambda;
    end
    
    ITIs(i) = ITI;  % Store the generated ITI
end

% Display the generated ITIs
disp('Generated ITIs:');
disp(ITIs);