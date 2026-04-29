% test_scaled_dataset.m
% Quick test script for the centralized scaled_dataset function

fprintf('========================================\n');
fprintf('Testing Centralized scaled_dataset.m\n');
fprintf('========================================\n\n');

% Change to root directory
cd /hdd/CNN_JC

% Track generated folders for cleanup
generated_folders = {};

%% Test 1: HN-GARCH with minimal parameters (fast test)
fprintf('TEST 1: HN-GARCH (small dataset)\n');
fprintf('------------------------------------\n');
try
    scaled_dataset('HN', ...
        'ContractsPerDay', 5, ...
        'NumDays', 2, ...
        'NumParamSets', 2);
    
    % Find the generated folder
    folders = dir('generation_hn_*');
    if ~isempty(folders)
        latest_folder = folders(end).name;
        generated_folders{end+1} = latest_folder;
        fprintf('✓ HN-GARCH test PASSED\n');
        fprintf('  Folder: %s\n', latest_folder);
    else
        fprintf('✗ Output folder not found\n');
    end
catch ME
    fprintf('✗ HN-GARCH test FAILED: %s\n', ME.message);
end
fprintf('\n');

%% Test 2: Duan GARCH with minimal parameters
fprintf('TEST 2: Duan GARCH (small dataset)\n');
fprintf('------------------------------------\n');
try
    scaled_dataset('Duan', ...
        'ContractsPerDay', 5, ...
        'NumDays', 2, ...
        'NumParamSets', 2);
    
    folders = dir('generation_duan_*');
    if ~isempty(folders)
        latest_folder = folders(end).name;
        generated_folders{end+1} = latest_folder;
        fprintf('✓ Duan GARCH test PASSED\n');
        fprintf('  Folder: %s\n', latest_folder);
    else
        fprintf('✗ Output folder not found\n');
    end
catch ME
    fprintf('✗ Duan GARCH test FAILED: %s\n', ME.message);
end
fprintf('\n');

%% Test 3: GJR-GARCH with minimal parameters
fprintf('TEST 3: GJR-GARCH (small dataset)\n');
fprintf('------------------------------------\n');
try
    scaled_dataset('GJR', ...
        'ContractsPerDay', 5, ...
        'NumDays', 2, ...
        'NumParamSets', 2);
    
    folders = dir('generation_gjr_*');
    if ~isempty(folders)
        latest_folder = folders(end).name;
        generated_folders{end+1} = latest_folder;
        fprintf('✓ GJR-GARCH test PASSED\n');
        fprintf('  Folder: %s\n', latest_folder);
    else
        fprintf('✗ Output folder not found\n');
    end
catch ME
    fprintf('✗ GJR-GARCH test FAILED: %s\n', ME.message);
end
fprintf('\n');

%% Verify output files in folders
fprintf('========================================\n');
fprintf('Verifying Output Folders\n');
fprintf('========================================\n');

for i = 1:length(generated_folders)
    folder = generated_folders{i};
    fprintf('\nFolder: %s\n', folder);
    
    % Check for required files
    files_to_check = {
        sprintf('%s/dataset_*.csv', folder), 'Dataset';
        sprintf('%s/garch_params.csv', folder), 'GARCH Params';
        sprintf('%s/asset_prices.csv', folder), 'Asset Prices';
        sprintf('%s/metadata.txt', folder), 'Metadata'
    };
    
    for j = 1:size(files_to_check, 1)
        pattern = files_to_check{j, 1};
        name = files_to_check{j, 2};
        found = dir(pattern);
        
        if ~isempty(found)
            file_info = dir(fullfile(folder, found(1).name));
            fprintf('  ✓ %s: %s (%.1f KB)\n', name, found(1).name, file_info.bytes/1024);
        else
            fprintf('  ✗ %s: NOT FOUND\n', name);
        end
    end
end

fprintf('\n========================================\n');
fprintf('All tests complete!\n');
fprintf('========================================\n');
fprintf('\nGenerated folders:\n');
for i = 1:length(generated_folders)
    fprintf('  • %s\n', generated_folders{i});
end
fprintf('\n');
