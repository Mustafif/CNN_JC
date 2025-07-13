% test_f_hhh.m
% Test script to verify f_hhh MEX file functionality

fprintf('Testing f_hhh MEX file functionality...\n');

% Add current directory to path
addpath(pwd);

% Check if f_hhh exists
fprintf('Checking if f_hhh exists: ');
if exist('f_hhh', 'file')
    fprintf('YES (type %d)\n', exist('f_hhh', 'file'));
else
    fprintf('NO\n');
    return;
end

% Test 1: Basic call with normal distribution parameters
fprintf('\nTest 1: Normal distribution (mu=0, sd=1, ka3=0, ka4=3)\n');
try
    [gamma, delta, xlam, xi, itype, ifault] = f_hhh(0, 1, 0, 3);
    fprintf('  SUCCESS: gamma=%.4f, delta=%.4f, xlam=%.4f, xi=%.4f, itype=%.0f, ifault=%.0f\n', ...
            gamma, delta, xlam, xi, itype, ifault);
catch err
    fprintf('  ERROR: %s\n', err.message);
end

% Test 2: Lognormal distribution
fprintf('\nTest 2: Lognormal distribution (mu=0.1, sd=0.2, ka3=0.5, ka4=2.5)\n');
try
    [gamma, delta, xlam, xi, itype, ifault] = f_hhh(0.1, 0.2, 0.5, 2.5);
    fprintf('  SUCCESS: gamma=%.4f, delta=%.4f, xlam=%.4f, xi=%.4f, itype=%.0f, ifault=%.0f\n', ...
            gamma, delta, xlam, xi, itype, ifault);
catch err
    fprintf('  ERROR: %s\n', err.message);
end

% Test 3: Edge case with zero standard deviation
fprintf('\nTest 3: Edge case (mu=1, sd=0, ka3=0, ka4=3)\n');
try
    [gamma, delta, xlam, xi, itype, ifault] = f_hhh(1, 0, 0, 3);
    fprintf('  SUCCESS: gamma=%.4f, delta=%.4f, xlam=%.4f, xi=%.4f, itype=%.0f, ifault=%.0f\n', ...
            gamma, delta, xlam, xi, itype, ifault);
catch err
    fprintf('  ERROR: %s\n', err.message);
end

% Test 4: Test with negative values
fprintf('\nTest 4: Negative values (mu=-0.1, sd=0.3, ka3=-0.2, ka4=4)\n');
try
    [gamma, delta, xlam, xi, itype, ifault] = f_hhh(-0.1, 0.3, -0.2, 4);
    fprintf('  SUCCESS: gamma=%.4f, delta=%.4f, xlam=%.4f, xi=%.4f, itype=%.0f, ifault=%.0f\n', ...
            gamma, delta, xlam, xi, itype, ifault);
catch err
    fprintf('  ERROR: %s\n', err.message);
end

% Test 5: Multiple calls to test stability
fprintf('\nTest 5: Multiple calls for stability test\n');
success_count = 0;
for i = 1:5
    try
        mu = 0.1 * i;
        sd = 0.2 + 0.1 * i;
        ka3 = 0.1 * (i - 3);
        ka4 = 3 + 0.5 * i;
        [gamma, delta, xlam, xi, itype, ifault] = f_hhh(mu, sd, ka3, ka4);
        success_count = success_count + 1;
    catch err
        fprintf('  Call %d failed: %s\n', i, err.message);
    end
end
fprintf('  SUCCESS: %d/5 calls completed successfully\n', success_count);

fprintf('\nf_hhh MEX file test completed.\n');
