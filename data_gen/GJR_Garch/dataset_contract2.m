function ds = dataset_contract2(garch)
    contract = [0.03 ...
    [5, 10, 21, 42, 63, 126, 180, 252, 360]...
    linspace(0.8, 1.2, 9)];
    alpha = garch(1);
    beta = garch(2);
    omega = garch(3);
    gamma = garch(4);
    lambda = garch(5);
    days_per_contract = 5;
    days = 1:days_per_contract;
    r = contract(1);
    T = contract(2:10);
    m = contract(11:19);
    ds = arrayfun(@(day) GenDaysData(days_per_contract, day, r, T, m, alpha, beta, omega, gamma, lambda), days, 'UniformOutput', false);
    ds = cell2mat(ds);
end