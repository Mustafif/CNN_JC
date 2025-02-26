function ds = dataset_contract(contract)
    alpha = 0.1;
    beta = 0.8;
    omega = 1e-6;
    gamma = 0;
    lambda = 0.15;
    days_per_contract = 31;
    days = 1:days_per_contract;
    r = contract(1);
    T = contract(2:10);
    m = contract(11:19);
    ds = arrayfun(@(day) GenDaysData(days_per_contract, day, r, T, m, alpha, beta, omega, gamma, lambda), days, 'UniformOutput', false);
    ds = cell2mat(ds);
end