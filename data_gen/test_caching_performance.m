% test_caching_performance.m
% Comprehensive test to validate extreme caching performance
% Tests all three versions and measures actual speedup

fprintf('=================================================================\n');
fprintf('         CACHING PERFORMANCE TEST - 50K-100K Datasets\n');
fprintf('=================================================================\n\n');

%% Configuration
test_sizes = [1000, 5000, 10000];  % Start small to avoid long tests
models = {'HN'};  % Test Heston-Nandi first

results = struct();

%% Test 1: Small dataset (1K contracts) - All versions
fprintf('TEST 1: Small Dataset (1K contracts)\n');
fprintf('-----------------------------------------------------------------\n');

contracts_per_day = 20;
num_days = 25;  % 20 × 25 × 2 = 1000 contracts

% Version 1: Basic scaled_dataset.m
fprintf('Running scaled_dataset.m (basic)...\n');
tic;
try
    scaled_dataset('HN', 'ContractsPerDay', contracts_per_day, ...
                  'NumDays', num_days, 'NumParamSets', 1);
    time_basic = toc;
    fprintf('  Time: %.2f seconds\n', time_basic);
    fprintf('  Throughput: %.0f contracts/sec\n', 1000/time_basic);
    results.basic_1k = time_basic;
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    time_basic = NaN;
    results.basic_1k = NaN;
end

% Version 2: scaled_dataset_optimized.m
fprintf('\nRunning scaled_dataset_optimized.m...\n');
tic;
try
    if exist('scaled_dataset_optimized.m', 'file')
        scaled_dataset_optimized('HN', 'ContractsPerDay', contracts_per_day, ...
                                'NumDays', num_days, 'NumParamSets', 1);
        time_optimized = toc;
        fprintf('  Time: %.2f seconds\n', time_optimized);
        fprintf('  Throughput: %.0f contracts/sec\n', 1000/time_optimized);
        fprintf('  Speedup vs basic: %.2fx\n', time_basic/time_optimized);
        results.optimized_1k = time_optimized;
    else
        fprintf('  File not found, skipping\n');
        time_optimized = NaN;
        results.optimized_1k = NaN;
    end
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    time_optimized = NaN;
    results.optimized_1k = NaN;
end

% Version 3: scaled_dataset_cached.m
fprintf('\nRunning scaled_dataset_cached.m (aggressive caching)...\n');
tic;
try
    if exist('scaled_dataset_cached.m', 'file')
        scaled_dataset_cached('HN', 'ContractsPerDay', contracts_per_day, ...
                             'NumDays', num_days, 'NumParamSets', 1, ...
                             'PrecomputeTrees', true, 'BatchSize', 500);
        time_cached = toc;
        fprintf('  Time: %.2f seconds\n', time_cached);
        fprintf('  Throughput: %.0f contracts/sec\n', 1000/time_cached);
        fprintf('  Speedup vs basic: %.2fx\n', time_basic/time_cached);
        fprintf('  Speedup vs optimized: %.2fx\n', time_optimized/time_cached);
        results.cached_1k = time_cached;
    else
        fprintf('  File not found, skipping\n');
        time_cached = NaN;
        results.cached_1k = NaN;
    end
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    time_cached = NaN;
    results.cached_1k = NaN;
end

fprintf('\n');

%% Test 2: Medium dataset (10K contracts)
fprintf('\nTEST 2: Medium Dataset (10K contracts)\n');
fprintf('-----------------------------------------------------------------\n');

contracts_per_day = 50;
num_days = 100;  % 50 × 100 × 2 = 10,000 contracts

fprintf('Running scaled_dataset_cached.m (10K)...\n');
tic;
try
    if exist('scaled_dataset_cached.m', 'file')
        scaled_dataset_cached('HN', 'ContractsPerDay', contracts_per_day, ...
                             'NumDays', num_days, 'NumParamSets', 1, ...
                             'PrecomputeTrees', true, 'BatchSize', 1000);
        time_cached_10k = toc;
        fprintf('  Time: %.2f seconds\n', time_cached_10k);
        fprintf('  Throughput: %.0f contracts/sec\n', 10000/time_cached_10k);
        results.cached_10k = time_cached_10k;
        
        % Estimate for larger datasets
        est_50k = time_cached_10k * 5.5;  % Slightly more than linear
        est_100k = time_cached_10k * 11.5;
        fprintf('  Estimated 50K:  %.2f seconds\n', est_50k);
        fprintf('  Estimated 100K: %.2f seconds\n', est_100k);
    else
        fprintf('  File not found, skipping\n');
        results.cached_10k = NaN;
    end
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    results.cached_10k = NaN;
end

fprintf('\n');

%% Test 3: Cache effectiveness
fprintf('\nTEST 3: Cache Effectiveness (Warm vs Cold)\n');
fprintf('-----------------------------------------------------------------\n');

contracts_per_day = 20;
num_days = 25;

fprintf('Cold run (building cache)...\n');
tic;
try
    if exist('scaled_dataset_cached.m', 'file')
        scaled_dataset_cached('HN', 'ContractsPerDay', contracts_per_day, ...
                             'NumDays', num_days, 'NumParamSets', 1, ...
                             'PrecomputeTrees', true, 'RebuildCache', true);
        time_cold = toc;
        fprintf('  Time: %.2f seconds\n', time_cold);
        results.cold_cache = time_cold;
    end
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    time_cold = NaN;
    results.cold_cache = NaN;
end

fprintf('\nWarm run (using cache)...\n');
tic;
try
    if exist('scaled_dataset_cached.m', 'file')
        scaled_dataset_cached('HN', 'ContractsPerDay', contracts_per_day, ...
                             'NumDays', num_days, 'NumParamSets', 1, ...
                             'PrecomputeTrees', true, 'RebuildCache', false);
        time_warm = toc;
        fprintf('  Time: %.2f seconds\n', time_warm);
        fprintf('  Cache speedup: %.2fx\n', time_cold/time_warm);
        results.warm_cache = time_warm;
    end
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    time_warm = NaN;
    results.warm_cache = NaN;
end

fprintf('\n');

%% Test 4: Multiple parameter sets (parallel processing)
fprintf('\nTEST 4: Multiple Parameter Sets (5 sets)\n');
fprintf('-----------------------------------------------------------------\n');

contracts_per_day = 10;
num_days = 10;
num_param_sets = 5;  % 10 × 10 × 2 × 5 = 1000 contracts total

fprintf('Running with %d parameter sets in parallel...\n', num_param_sets);
tic;
try
    if exist('scaled_dataset_cached.m', 'file')
        scaled_dataset_cached('HN', 'ContractsPerDay', contracts_per_day, ...
                             'NumDays', num_days, 'NumParamSets', num_param_sets, ...
                             'PrecomputeTrees', true);
        time_multi = toc;
        fprintf('  Time: %.2f seconds\n', time_multi);
        fprintf('  Throughput: %.0f contracts/sec\n', 1000/time_multi);
        results.multi_param = time_multi;
    end
catch ME
    fprintf('  ERROR: %s\n', ME.message);
    results.multi_param = NaN;
end

fprintf('\n');

%% Summary
fprintf('\n=================================================================\n');
fprintf('                     PERFORMANCE SUMMARY\n');
fprintf('=================================================================\n\n');

fprintf('1K Contracts:\n');
fprintf('  Basic:          %s\n', format_time(results.basic_1k));
fprintf('  Optimized:      %s\n', format_time(results.optimized_1k));
fprintf('  Cached:         %s\n', format_time(results.cached_1k));
if ~isnan(results.basic_1k) && ~isnan(results.cached_1k)
    fprintf('  Total Speedup:  %.2fx\n', results.basic_1k/results.cached_1k);
end

fprintf('\n10K Contracts:\n');
fprintf('  Cached:         %s\n', format_time(results.cached_10k));
if ~isnan(results.cached_10k)
    fprintf('  Throughput:     %.0f/sec\n', 10000/results.cached_10k);
    fprintf('  Est. 50K:       %.2f seconds\n', results.cached_10k * 5.5);
    fprintf('  Est. 100K:      %.2f seconds\n', results.cached_10k * 11.5);
end

fprintf('\nCache Effectiveness:\n');
fprintf('  Cold:           %s\n', format_time(results.cold_cache));
fprintf('  Warm:           %s\n', format_time(results.warm_cache));
if ~isnan(results.cold_cache) && ~isnan(results.warm_cache)
    fprintf('  Cache Speedup:  %.2fx\n', results.cold_cache/results.warm_cache);
end

fprintf('\nMulti-Parameter:\n');
fprintf('  5 param sets:   %s\n', format_time(results.multi_param));

fprintf('\n=================================================================\n');
fprintf('                    TARGET VALIDATION\n');
fprintf('=================================================================\n\n');

if ~isnan(results.cached_10k)
    est_100k = results.cached_10k * 11.5;
    
    fprintf('Target: 100K contracts in < 60 seconds\n');
    if est_100k < 60
        fprintf('Status: ✓ ACHIEVED (est. %.2f sec)\n', est_100k);
    else
        fprintf('Status: ✗ MISSED (est. %.2f sec)\n', est_100k);
        fprintf('Need: %.2fx additional speedup\n', est_100k/60);
    end
    
    fprintf('\nTarget: 50K contracts in < 30 seconds\n');
    est_50k = results.cached_10k * 5.5;
    if est_50k < 30
        fprintf('Status: ✓ ACHIEVED (est. %.2f sec)\n', est_50k);
    else
        fprintf('Status: ✗ MISSED (est. %.2f sec)\n', est_50k);
        fprintf('Need: %.2fx additional speedup\n', est_50k/30);
    end
else
    fprintf('Cannot validate - 10K test failed\n');
end

fprintf('\n=================================================================\n');
fprintf('                 RECOMMENDATIONS\n');
fprintf('=================================================================\n\n');

if ~isnan(results.cached_10k)
    throughput = 10000 / results.cached_10k;
    
    if throughput < 1000
        fprintf('⚠ Low throughput (%.0f/sec)\n', throughput);
        fprintf('  Suggestions:\n');
        fprintf('  • Increase BatchSize to 1000\n');
        fprintf('  • Ensure PrecomputeTrees=true\n');
        fprintf('  • Check CPU usage (should be ~100%%)\n');
        fprintf('  • Consider vectorized IV solver\n');
    elseif throughput < 2000
        fprintf('✓ Good throughput (%.0f/sec)\n', throughput);
        fprintf('  To improve further:\n');
        fprintf('  • Implement vectorized IV solver (2-3x boost)\n');
        fprintf('  • Use persistent disk cache\n');
        fprintf('  • Consider GPU acceleration\n');
    else
        fprintf('✓✓ Excellent throughput (%.0f/sec)\n', throughput);
        fprintf('  Achieving target performance!\n');
        fprintf('  Optional enhancements:\n');
        fprintf('  • GPU acceleration for ultimate speed\n');
        fprintf('  • Persistent cache for repeated runs\n');
    end
end

fprintf('\n=================================================================\n');
fprintf('                      TEST COMPLETE\n');
fprintf('=================================================================\n');

% Save results
save('caching_test_results.mat', 'results');
fprintf('\nResults saved to: caching_test_results.mat\n');

%% Helper function
function str = format_time(t)
    if isnan(t)
        str = 'N/A';
    elseif t < 1
        str = sprintf('%.3f sec', t);
    else
        str = sprintf('%.2f sec', t);
    end
end
