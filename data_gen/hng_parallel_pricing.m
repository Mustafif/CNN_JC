
% Heston-Nandi GARCH Option Pricing (Parallel Compatible Version)
% Author: Refactored for parallelism
% Description: Parallelizes over each day's dataset

clear; clc;

% Load data
load('GARCHparams.mat', 'params_matrix');  % [date_idx x params]
load('contracts_data.mat', 'contract_data');  % struct array with per-day data
output_dir = 'output_parallel_days';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Constants
r = 0.01;
option_type = 1;  % 1 = call, -1 = put
tree_steps = 300;
num_days = numel(contract_data);

% Parallel processing per day
parfor day = 1:num_days
    fprintf("Processing day %d/%d...\n", day, num_days);

    % Extract parameters for the day
    params = params_matrix(day, :);
    omega = params(1); alpha = params(2); beta = params(3);
    gamma = params(4); lambda = params(5); h_0 = params(6);

    contracts = contract_data(day).contracts;
    n_contracts = size(contracts, 1);

    % Preallocate result matrix: [S0, K, tau, price, impvol]
    results = NaN(n_contracts, 5);

    for c = 1:n_contracts
        S0 = contracts(c, 1);
        K = contracts(c, 2);
        tau = contracts(c, 3);
        h = h_0;

        % Compute option price via HNG model
        price = hng_qe_mle(S0, K, r, tau, omega, alpha, beta, gamma, lambda, h, tree_steps, option_type);

        % Implied volatility
        impvol = blsimpv(S0, K, r, tau/252, price, [], [], [], option_type == 1);

        % Store result
        results(c, :) = [S0, K, tau, price, impvol];
    end

    % Save CSV
    filename = fullfile(output_dir, sprintf('options_day_%03d.csv', day));
    header = 'S0,K,tau,price,impvol';
    writematrix(header, filename);
    dlmwrite(filename, results, '-append');
end
