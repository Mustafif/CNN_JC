% Number of contracts to generate 
n = 40;
days_per_contract = 5;

garch_params = genDuanGarchParams(n);
market_params = genMarketParams(n);

contracts = arrayfun(@(i) dataset_contract3(garch_params(i, :), market_params(i, :), days_per_contract), 1:n, 'UniformOutput', false);
ds = [contracts{:}];

headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'V'}';
filename = 'stage3.csv';
dataset = [headers'; num2cell(ds')];
writecell(dataset, filename);
disp(['Dataset saved as ', filename]);