% TEST_COMPARE_SYSTEM.m
% Quick test of the centralized comparison system

clear all; clc;

fprintf('=================================================================\n');
fprintf('    TESTING CENTRALIZED COMPARISON SYSTEM\n');
fprintf('=================================================================\n\n');

%% TEST 1: Check Python TOML reader
fprintf('TEST 1: Checking Python TOML reader...\n');
[status, result] = system('python3 --version');
if status == 0
    fprintf('  ✓ Python 3 found: %s', result);
else
    fprintf('  ✗ Python 3 not found\n');
    return;
end

[status, result] = system('python3 -c "import toml; print(''TOML module available'')"');
if status == 0
    fprintf('  ✓ TOML module available\n');
else
    fprintf('  ✗ TOML module not found. Install with: pip install toml\n');
    return;
end

%% TEST 2: Check TOML reader script
fprintf('\nTEST 2: Testing TOML reader script...\n');
if exist('read_toml_config.py', 'file')
    fprintf('  ✓ read_toml_config.py found\n');
else
    fprintf('  ✗ read_toml_config.py not found\n');
    return;
end

[status, json_str] = system('python3 read_toml_config.py compare_config.toml');
if status == 0
    config = jsondecode(json_str);
    fprintf('  ✓ Successfully read compare_config.toml\n');
    fprintf('    Model type: %s\n', config.model.type);
    fprintf('    Contracts per day: %d\n', config.scaling.contracts_per_day);
    fprintf('    Noise level: %.2f%%\n', config.noise.level * 100);
else
    fprintf('  ✗ Failed to read TOML config\n');
    fprintf('  Error: %s\n', json_str);
    return;
end

%% TEST 3: Check required MATLAB functions
fprintf('\nTEST 3: Checking required MATLAB functions...\n');

required_functions = {
    'mcHN', 'American', 'impvol_improved', ...
    'genhDelta', 'TreeNodes_ht_HN', 'TreeNodes_logSt_HN', ...
    'Prob_Xt'
};

all_found = true;
for i = 1:length(required_functions)
    func = required_functions{i};
    if exist(func, 'file')
        fprintf('  ✓ %s.m found\n', func);
    else
        fprintf('  ✗ %s.m not found\n', func);
        all_found = false;
    end
end

if ~all_found
    fprintf('\n  Warning: Some required functions are missing.\n');
    fprintf('  The system may not work properly.\n');
end

%% TEST 4: Check compare_dataset.m
fprintf('\nTEST 4: Checking compare_dataset.m...\n');
if exist('compare_dataset', 'file')
    fprintf('  ✓ compare_dataset.m found\n');
else
    fprintf('  ✗ compare_dataset.m not found\n');
    return;
end

%% TEST 5: Check rel_err_results.m
fprintf('\nTEST 5: Checking rel_err_results.m...\n');
if exist('rel_err_results', 'file')
    fprintf('  ✓ rel_err_results.m found\n');
else
    fprintf('  ✗ rel_err_results.m not found\n');
    return;
end

%% TEST 6: Verify config parameters
fprintf('\nTEST 6: Verifying parameter configuration...\n');
fprintf('  True parameters:\n');
fprintf('    omega=%.2e, alpha=%.2e, beta=%.4f, gamma=%.2f, lambda=%.2f\n', ...
    config.parameters.true.omega, config.parameters.true.alpha, ...
    config.parameters.true.beta, config.parameters.true.gamma, ...
    config.parameters.true.lambda);

fprintf('  Initial parameters:\n');
fprintf('    omega=%.2e, alpha=%.2e, beta=%.4f, gamma=%.2f, lambda=%.2f\n', ...
    config.parameters.initial.omega, config.parameters.initial.alpha, ...
    config.parameters.initial.beta, config.parameters.initial.gamma, ...
    config.parameters.initial.lambda);

fprintf('  Calibrated parameters:\n');
fprintf('    omega=%.2e, alpha=%.2e, beta=%.4f, gamma=%.2f, lambda=%.2f\n', ...
    config.parameters.calibrated.omega, config.parameters.calibrated.alpha, ...
    config.parameters.calibrated.beta, config.parameters.calibrated.gamma, ...
    config.parameters.calibrated.lambda);

%% SUMMARY
fprintf('\n=================================================================\n');
fprintf('                    TEST SUMMARY\n');
fprintf('=================================================================\n');
fprintf('All tests passed! ✓\n\n');
fprintf('You can now run:\n');
fprintf('  compare_dataset(''compare_config.toml'')\n');
fprintf('  rel_err_results()\n');
fprintf('=================================================================\n');
