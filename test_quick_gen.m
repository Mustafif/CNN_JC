% Quick test of scaled_dataset with small parameters
cd /hdd/CNN_JC

fprintf('Testing scaled_dataset with minimal parameters...\n\n');

try
    scaled_dataset('HN', 'ContractsPerDay', 3, 'NumDays', 2, 'NumParamSets', 1);
    fprintf('\n✓ Generation succeeded!\n');
catch ME
    fprintf('\n✗ Generation failed!\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
