function ds = dataset_contract3(garch, market, days_per_contract)
    alpha = garch(1);
    beta = garch(2);
    omega = garch(3);
    theta = garch(4);
    lambda = garch(5);
    m = market(1:9);
    r = market(10);
    T = market(11:19);
    days = 1:days_per_contract;
    ds = arrayfun(@(day) GenDaysData(days_per_contract, day, r, T, m, alpha, beta, omega, theta, lambda), days, 'UniformOutput', false);
    ds = cell2mat(ds);
end