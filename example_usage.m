% example_usage.m
% Examples demonstrating different use cases of scaled_dataset.m

fprintf('=================================================================\n');
fprintf('        SCALED_DATASET.M - USAGE EXAMPLES\n');
fprintf('=================================================================\n\n');

% Make sure we're in the correct directory
cd /hdd/CNN_JC

%% EXAMPLE 1: Quick Start - Default Parameters
% Generates a small dataset with default settings
fprintf('EXAMPLE 1: Quick start with defaults\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Command: scaled_dataset(''HN'')\n');
fprintf('This will create a timestamped folder with:\n');
fprintf('  - 50 contracts per day × 20 days × 5 param sets\n');
fprintf('  - Cleaned dataset (V >= 0.5, T != 1/252)\n');
fprintf('  - GARCH parameters file\n');
fprintf('  - Asset prices file\n');
fprintf('  - Metadata file\n');
fprintf('  - Output: generation_hn_<timestamp>/\n\n');

% Uncomment to run:
% scaled_dataset('HN')

%% EXAMPLE 2: Medium-Sized Dataset for Training
fprintf('EXAMPLE 2: Medium dataset for ML training\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Command:\n');
fprintf('  scaled_dataset(''Duan'', ...\n');
fprintf('      ''ContractsPerDay'', 100, ...\n');
fprintf('      ''NumDays'', 50, ...\n');
fprintf('      ''NumParamSets'', 10)\n\n');
fprintf('This generates: ~10,000 contracts (after cleaning)\n');
fprintf('  - Diverse GARCH parameters (10 different sets)\n');
fprintf('  - Good variety of spot prices (50 days)\n');
fprintf('  - Suitable for CNN training\n');
fprintf('  - Output folder contains all files needed\n\n');

% Uncomment to run:
% scaled_dataset('Duan', 'ContractsPerDay', 100, 'NumDays', 50, 'NumParamSets', 10)

%% EXAMPLE 3: Large Dataset with Custom Ranges
fprintf('EXAMPLE 3: Large dataset with wide moneyness/maturity ranges\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Command:\n');
fprintf('  scaled_dataset(''GJR'', ...\n');
fprintf('      ''ContractsPerDay'', 250, ...\n');
fprintf('      ''NumDays'', 60, ...\n');
fprintf('      ''NumParamSets'', 15, ...\n');
fprintf('      ''MoneynessRange'', [0.7, 1.3], ...\n');
fprintf('      ''MaturityRange'', [0.05, 2.0])\n\n');
fprintf('This generates: 30,000 contracts\n');
fprintf('  - Wide moneyness range (0.7 to 1.3)\n');
fprintf('  - Long maturities (up to 2 years)\n');
fprintf('  - 15 different GARCH parameter sets\n\n');

% Uncomment to run:
scaled_dataset('GJR', 'ContractsPerDay', 250, 'NumDays', 60, 'NumParamSets', 15, ...
               'MoneynessRange', [0.7, 1.3], 'MaturityRange', [0.05, 2.0])

%% EXAMPLE 4: Comparison Across Models
fprintf('EXAMPLE 4: Generate datasets for all three models (for comparison)\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Commands:\n');
fprintf('  scaled_dataset(''HN'',   ''ContractsPerDay'', 50, ''NumDays'', 20, ''OutputFile'', ''compare_hn.csv'')\n');
fprintf('  scaled_dataset(''Duan'', ''ContractsPerDay'', 50, ''NumDays'', 20, ''OutputFile'', ''compare_duan.csv'')\n');
fprintf('  scaled_dataset(''GJR'',  ''ContractsPerDay'', 50, ''NumDays'', 20, ''OutputFile'', ''compare_gjr.csv'')\n\n');
fprintf('Use this to compare model behavior with identical contract specifications\n\n');

% Uncomment to run:
% scaled_dataset('HN',   'ContractsPerDay', 50, 'NumDays', 20, 'OutputFile', 'compare_hn.csv')
% scaled_dataset('Duan', 'ContractsPerDay', 50, 'NumDays', 20, 'OutputFile', 'compare_duan.csv')
% scaled_dataset('GJR',  'ContractsPerDay', 50, 'NumDays', 20, 'OutputFile', 'compare_gjr.csv')

%% EXAMPLE 5: Small Test Dataset (for debugging)
fprintf('EXAMPLE 5: Tiny test dataset (for quick testing/debugging)\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Command:\n');
fprintf('  scaled_dataset(''HN'', ...\n');
fprintf('      ''ContractsPerDay'', 10, ...\n');
fprintf('      ''NumDays'', 5, ...\n');
fprintf('      ''NumParamSets'', 2, ...\n');
fprintf('      ''BatchSize'', 5, ...\n');
fprintf('      ''OutputFile'', ''debug_test.csv'')\n\n');
fprintf('This generates: 100 contracts (very fast)\n');
fprintf('  - Useful for testing code changes\n');
fprintf('  - Quick validation of pipeline\n\n');

% Uncomment to run:
% scaled_dataset('HN', 'ContractsPerDay', 10, 'NumDays', 5, 'NumParamSets', 2, ...
%                'BatchSize', 5, 'OutputFile', 'debug_test.csv')

%% EXAMPLE 6: Production-Scale Dataset
fprintf('EXAMPLE 6: Large production dataset\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Command:\n');
fprintf('  scaled_dataset(''Duan'', ...\n');
fprintf('      ''ContractsPerDay'', 500, ...\n');
fprintf('      ''NumDays'', 100, ...\n');
fprintf('      ''NumParamSets'', 20, ...\n');
fprintf('      ''BatchSize'', 50, ...\n');
fprintf('      ''OutputFile'', ''production_duan_100k.csv'')\n\n');
fprintf('This generates: 100,000 contracts\n');
fprintf('  - Sufficient for deep learning models\n');
fprintf('  - 20 diverse GARCH parameter sets\n');
fprintf('  - Batch processing to manage memory\n');
fprintf('  - WARNING: This will take several hours!\n\n');

% Uncomment to run (WARNING: SLOW!):
% scaled_dataset('Duan', 'ContractsPerDay', 500, 'NumDays', 100, 'NumParamSets', 20, ...
%                'BatchSize', 50, 'OutputFile', 'production_duan_100k.csv')

%% EXAMPLE 7: At-The-Money Options Only
fprintf('EXAMPLE 7: Focus on at-the-money options\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Command:\n');
fprintf('  scaled_dataset(''HN'', ...\n');
fprintf('      ''ContractsPerDay'', 100, ...\n');
fprintf('      ''NumDays'', 30, ...\n');
fprintf('      ''MoneynessRange'', [0.95, 1.05])\n\n');
fprintf('This generates: 6,000 near-ATM contracts\n');
fprintf('  - Tight moneyness range around ATM\n');
fprintf('  - Good for studying volatility smile behavior\n\n');

% Uncomment to run:
% scaled_dataset('HN', 'ContractsPerDay', 100, 'NumDays', 30, 'MoneynessRange', [0.95, 1.05])

%% EXAMPLE 8: Short-Term Options
fprintf('EXAMPLE 8: Short-dated options only\n');
fprintf('-----------------------------------------------------------------\n');
fprintf('Command:\n');
fprintf('  scaled_dataset(''GJR'', ...\n');
fprintf('      ''ContractsPerDay'', 150, ...\n');
fprintf('      ''NumDays'', 40, ...\n');
fprintf('      ''MaturityRange'', [0.02, 0.25])\n\n');
fprintf('This generates: 12,000 short-term contracts\n');
fprintf('  - Maturities from 1 week to 3 months\n');
fprintf('  - Useful for studying short-term dynamics\n\n');

% Uncomment to run:
% scaled_dataset('GJR', 'ContractsPerDay', 150, 'NumDays', 40, 'MaturityRange', [0.02, 0.25])

%% Summary
fprintf('=================================================================\n');
fprintf('                          SUMMARY\n');
fprintf('=================================================================\n');
fprintf('To run any example, uncomment the corresponding line(s) and execute.\n');
fprintf('\nRecommended starting point:\n');
fprintf('  - Example 2 for ML training (10K contracts, ~30 minutes)\n');
fprintf('  - Example 5 for testing (100 contracts, ~1 minute)\n');
fprintf('\nFor documentation, see: DATASET_GENERATOR_README.md\n');
fprintf('=================================================================\n\n');
