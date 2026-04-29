% Quick test of the fixed mcHN function
cd /hdd/CNN_JC
addpath(genpath('./data_gen'));

fprintf('Testing mcHN fix...\n');

% Parameters
M = 5;
N = 10;
S0 = 100;
r = 0.05/252;
omega = 8e-7;
alpha = 1.25e-6;
beta = 0.78;
gamma = 95;
lambda = 0.5;

% Generate Z matrix
numPoint = N + 1;
Z = randn(numPoint + 1, M);

% Test mcHN
try
    [S_paths, h0] = mcHN(M, N, S0, Z, r, omega, alpha, beta, gamma, lambda);
    fprintf('✓ mcHN succeeded!\n');
    fprintf('  Output dimensions: %dx%d\n', size(S_paths, 1), size(S_paths, 2));
    fprintf('  Initial variance h0: %.6f\n', h0);
    fprintf('  Final prices: ');
    fprintf('%.2f ', S_paths(end, :));
    fprintf('\n');
catch ME
    fprintf('✗ mcHN failed: %s\n', ME.message);
end

fprintf('\nTest complete!\n');
