% BENCHMARK_PERFORMANCE.m
% Benchmark scaled_dataset.m vs scaled_dataset_optimized.m

clear all; clc;

fprintf('=================================================================\n');
fprintf('        PERFORMANCE BENCHMARK: DATASET GENERATION\n');
fprintf('=================================================================\n\n');

%% TEST CONFIGURATION
% Small test to compare performance
CONTRACTS_PER_DAY = 20;
NUM_DAYS = 5;
NUM_PARAM_SETS = 3;

fprintf('BENCHMARK CONFIGURATION:\n');
fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('  Number of days: %d\n', NUM_DAYS);
fprintf('  Parameter sets: %d\n', NUM_PARAM_SETS);
fprintf('  Total contracts: %d\n\n', CONTRACTS_PER_DAY * NUM_DAYS * 2 * NUM_PARAM_SETS);

%% CHECK PARALLEL POOL
pool = gcp('nocreate');
if isempty(pool)
    fprintf('Starting parallel pool...\n');
    try
        parpool(feature('numcores'));
        fprintf('✓ Parallel pool started with %d workers\n\n', feature('numcores'));
    catch
        fprintf('⚠ Could not start parallel pool\n\n');
    end
else
    fprintf('✓ Using existing parallel pool (%d workers)\n\n', pool.NumWorkers);
end

%% BENCHMARK 1: Current scaled_dataset.m
fprintf('=================================================================\n');
fprintf('BENCHMARK 1: scaled_dataset.m (Current Implementation)\n');
fprintf('=================================================================\n');

if exist('scaled_dataset', 'file')
    try
        fprintf('Running...\n');
        tic;
        scaled_dataset('HN', 'ContractsPerDay', CONTRACTS_PER_DAY, ...
                            'NumDays', NUM_DAYS, ...
                            'NumParamSets', NUM_PARAM_SETS);
        time_current = toc;
        
        fprintf('\n✓ Current implementation complete\n');
        fprintf('  Time elapsed: %.2f seconds\n', time_current);
        fprintf('  Throughput: %.1f contracts/second\n', ...
                CONTRACTS_PER_DAY * NUM_DAYS * 2 * NUM_PARAM_SETS / time_current);
        
    catch ME
        fprintf('✗ Error: %s\n', ME.message);
        time_current = NaN;
    end
else
    fprintf('⚠ scaled_dataset.m not found, skipping\n');
    time_current = NaN;
end

fprintf('\n');
pause(2);  % Brief pause between tests

%% BENCHMARK 2: Optimized scaled_dataset_optimized.m
fprintf('=================================================================\n');
fprintf('BENCHMARK 2: scaled_dataset_optimized.m (New Implementation)\n');
fprintf('=================================================================\n');

if exist('scaled_dataset_optimized', 'file')
    try
        fprintf('Running...\n');
        tic;
        scaled_dataset_optimized('HN', 'ContractsPerDay', CONTRACTS_PER_DAY, ...
                                      'NumDays', NUM_DAYS, ...
                                      'NumParamSets', NUM_PARAM_SETS);
        time_optimized = toc;
        
        fprintf('\n✓ Optimized implementation complete\n');
        fprintf('  Time elapsed: %.2f seconds\n', time_optimized);
        fprintf('  Throughput: %.1f contracts/second\n', ...
                CONTRACTS_PER_DAY * NUM_DAYS * 2 * NUM_PARAM_SETS / time_optimized);
        
    catch ME
        fprintf('✗ Error: %s\n', ME.message);
        time_optimized = NaN;
    end
else
    fprintf('⚠ scaled_dataset_optimized.m not found, skipping\n');
    time_optimized = NaN;
end

%% COMPARISON
fprintf('\n=================================================================\n');
fprintf('                    PERFORMANCE COMPARISON\n');
fprintf('=================================================================\n');

if ~isnan(time_current) && ~isnan(time_optimized)
    speedup = time_current / time_optimized;
    
    fprintf('Current implementation:   %.2f seconds\n', time_current);
    fprintf('Optimized implementation: %.2f seconds\n', time_optimized);
    fprintf('───────────────────────────────────────\n');
    fprintf('Speedup: %.2fx\n', speedup);
    fprintf('Time saved: %.2f seconds (%.1f%%)\n', time_current - time_optimized, ...
            100 * (time_current - time_optimized) / time_current);
    
    if speedup > 2
        fprintf('\n🚀 Excellent! >2x speedup achieved\n');
    elseif speedup > 1.5
        fprintf('\n✓ Good! >1.5x speedup achieved\n');
    elseif speedup > 1.1
        fprintf('\n✓ Moderate improvement\n');
    else
        fprintf('\n⚠ Limited improvement (may be due to overhead on small dataset)\n');
    end
    
    % Estimate for larger datasets
    fprintf('\nESTIMATED PERFORMANCE FOR LARGER DATASETS:\n');
    fprintf('  10,000 contracts: Current=%.1fmin, Optimized=%.1fmin\n', ...
            time_current * 10000 / (CONTRACTS_PER_DAY * NUM_DAYS * 2 * NUM_PARAM_SETS) / 60, ...
            time_optimized * 10000 / (CONTRACTS_PER_DAY * NUM_DAYS * 2 * NUM_PARAM_SETS) / 60);
    fprintf('  100,000 contracts: Current=%.1fmin, Optimized=%.1fmin\n', ...
            time_current * 100000 / (CONTRACTS_PER_DAY * NUM_DAYS * 2 * NUM_PARAM_SETS) / 60, ...
            time_optimized * 100000 / (CONTRACTS_PER_DAY * NUM_DAYS * 2 * NUM_PARAM_SETS) / 60);
    
elseif ~isnan(time_current)
    fprintf('Only current implementation available\n');
    fprintf('Time: %.2f seconds\n', time_current);
elseif ~isnan(time_optimized)
    fprintf('Only optimized implementation available\n');
    fprintf('Time: %.2f seconds\n', time_optimized);
else
    fprintf('No benchmarks completed successfully\n');
end

%% SYSTEM INFORMATION
fprintf('\n=================================================================\n');
fprintf('                    SYSTEM INFORMATION\n');
fprintf('=================================================================\n');
fprintf('MATLAB version: %s\n', version);
fprintf('CPU cores: %d\n', feature('numcores'));
fprintf('Parallel workers: %d\n', pool.NumWorkers);

% Check GPU
try
    gpu = gpuDevice();
    fprintf('GPU: %s (%.1f GB)\n', gpu.Name, gpu.AvailableMemory / 1e9);
catch
    fprintf('GPU: Not available\n');
end

% Memory info
if ispc
    [~, sys] = memory;
    fprintf('System RAM: %.1f GB\n', sys.PhysicalMemory.Total / 1e9);
    fprintf('Available RAM: %.1f GB\n', sys.PhysicalMemory.Available / 1e9);
end

fprintf('=================================================================\n');

%% RECOMMENDATIONS
fprintf('\nRECOMMENDATIONS:\n');

if ~isnan(time_current) && ~isnan(time_optimized)
    if speedup > 1.5
        fprintf('✓ Use scaled_dataset_optimized.m for production\n');
        fprintf('✓ Expected speedup scales with dataset size\n');
        fprintf('✓ Consider increasing batch size for even better performance\n');
    else
        fprintf('⚠ Speedup is modest for small datasets\n');
        fprintf('⚠ Try larger datasets to see more benefit\n');
        fprintf('⚠ For small datasets, either version is fine\n');
    end
end

fprintf('\nTo test with larger datasets:\n');
fprintf('  benchmark_performance_large  %% Run extended benchmark\n');

fprintf('\n=================================================================\n');
