% quick_test_cached.m
% Quick syntax test for scaled_dataset_cached.m

fprintf('=================================================================\n');
fprintf('       Quick Syntax Test for scaled_dataset_cached.m\n');
fprintf('=================================================================\n\n');

try
    % Very small test (100 contracts)
    fprintf('Running small test (100 contracts)...\n');
    tic;
    scaled_dataset_cached('HN', ...
        'ContractsPerDay', 5, ...
        'NumDays', 10, ...          % 5 × 10 × 2 = 100 contracts
        'NumParamSets', 1, ...
        'PrecomputeTrees', true);
    time_elapsed = toc;
    
    fprintf('\n');
    fprintf('SUCCESS! ✓\n');
    fprintf('Time: %.2f seconds\n', time_elapsed);
    fprintf('Throughput: %.0f contracts/sec\n', 100/time_elapsed);
    fprintf('\nThe file is working correctly!\n');
    
    % Estimate for larger datasets
    throughput = 100/time_elapsed;
    est_100k = 100000 / throughput;
    fprintf('\nEstimated performance for 100K contracts: %.2f seconds\n', est_100k);
    
    if est_100k < 60
        fprintf('Status: ✓ Should meet <60 second target!\n');
    else
        fprintf('Status: ⚠ May need optimization\n');
    end
    
catch ME
    fprintf('\n');
    fprintf('ERROR ✗\n');
    fprintf('Message: %s\n', ME.message);
    fprintf('File: %s\n', ME.stack(1).file);
    fprintf('Line: %d\n', ME.stack(1).line);
    
    % Print full error for debugging
    fprintf('\nFull error:\n');
    disp(getReport(ME));
end

fprintf('\n=================================================================\n');
