r = 0.05;
T = [5, 13, 25, 42, 63, 126, 180, 252, 360];
m = linspace(0.8, 1.2, 9);
alpha = 1.33e-6;
beta = 0.8;
omega = 1e-6;
gamma = 100;
lambda = 0.5;

% Use array to store days
max_day = 13;
days = 1:max_day;

% Preallocate dataset generation
dataset = arrayfun(@(day) GenDaysData(max_day, day, r, T, m, alpha, beta, omega, gamma, lambda), days, 'UniformOutput', false);
dataset = cell2mat(dataset);

% Prepare headers and save dataset
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'V'}';
filename = "test_dataset.csv";

% Combine headers with data and write to CSV
full_dataset = [headers'; num2cell(dataset')];
writecell(full_dataset, filename);

% Confirmation
disp(['Dataset saved as ', filename]);