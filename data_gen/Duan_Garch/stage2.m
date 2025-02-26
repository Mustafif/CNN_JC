% Set the number of contracts
n = 27;

% Generate n sets of HNGARCH parameters
params = genDuanGarchParams(n);

% Generate datasets for each contract
contracts = arrayfun(@(i) dataset_contract2(params(i, :)), 1:n, 'UniformOutput', false);

% Concatenate datasets
ds = [contracts{:}];

% Define headers for dataset
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'V'}';

% Save dataset to CSV
filename = 'stage2.csv';
dataset = [headers'; num2cell(ds')];
writecell(dataset, filename);
disp(['Dataset saved as ', filename]);