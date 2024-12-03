% r = 0.05;
% T = [5, 15, 21, 45, 63, 126];
% m = linspace(0.45, 1.5, 9);
% Original 
r = 0.03;
T = [5, 10, 21, 42, 63, 126];
m = linspace(0.8, 1.2, 9);

alpha = 1.33e-6;
beta = 0.8;
omega = 1e-6;
gamma = 100;
lambda = 0.5;

day1 = GenDaysData(1, r, T, m, alpha, beta, omega, gamma, lambda);
day2 = GenDaysData(2, r, T, m, alpha, beta, omega, gamma, lambda);
day3 = GenDaysData(3, r, T, m, alpha, beta, omega, gamma, lambda);
day4 = GenDaysData(4, r, T, m, alpha, beta, omega, gamma, lambda);
day5 = GenDaysData(5, r, T, m, alpha, beta, omega, gamma, lambda);

dataset = [day1 day2 day3 day4 day5];
headers = {'S0', 'K', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'V'};

% Define the filename
filename = "train_dataset.csv";
dataset  = [headers' num2cell(dataset) ];
writecell(dataset, filename);

% Confirmation
disp(['Dataset saved as ', filename]);