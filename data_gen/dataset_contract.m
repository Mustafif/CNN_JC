function ds = dataset_contract(contract)
    % Base GARCH parameters - more realistic for generating volatility smile
    alpha_base = 0.05;    % Increased from 1.33e-6
    beta = 0.8;
    omega_base = 0.00001; % Increased from 1e-6
    gamma_base = 1.5;     % Decreased from 100 to be more realistic
    lambda_base = 0.3;    % Adjusted from 0.5 to introduce asymmetry
    
    days_per_contract = 31;
    days = 1:days_per_contract;
    r = contract(1);
    T = contract(2:10);
    m = contract(11:19);
    
    % Create parameter arrays that vary with moneyness to produce smile effect
    moneyness_range = max(m) - min(m);
    mid_moneyness = (max(m) + min(m))/2;
    
    % Parameters for each moneyness level
    alpha_params = zeros(1, length(m));
    omega_params = zeros(1, length(m));
    gamma_params = zeros(1, length(m));
    lambda_params = zeros(1, length(m));
    
    for i = 1:length(m)
        % Create smile effect by varying parameters based on distance from ATM
        distance = abs(m(i) - mid_moneyness);
        smile_factor = 1 + 3 * (distance / (moneyness_range/2))^2;
        
        % Adjust parameters to create volatility smile
        alpha_params(i) = alpha_base * smile_factor;
        omega_params(i) = omega_base * smile_factor;
        gamma_params(i) = gamma_base * (1 + 0.5 * (m(i) - mid_moneyness)); % Skew
        lambda_params(i) = lambda_base * (1 - 0.2 * (m(i) - mid_moneyness)); % Opposite skew
    end
    
    ds = arrayfun(@(day) GenDaysDataSmile(days_per_contract, day, r, T, m, alpha_params, beta, omega_params, gamma_params, lambda_params), days, 'UniformOutput', false);
    ds = cell2mat(ds);
end

function dataset = GenDaysDataSmile(max_day, day_num, r, T, m, alpha_params, beta, omega_params, gamma_params, lambda_params)
    % Modified version of GenDaysData that accepts parameter arrays
    % Generate dataset for a given day number using market and GARCH parameters.
    
    % Simulation parameters
    N = 15; % Increased from 5 to 15 for better accuracy
    M = 1;   % Number of Monte Carlo paths
    S_init = 100; % Initial stock price

    % Generate random standard normal samples
    Z = randn(N + 1, M);

    % Use parameters for mid-moneyness point for stock generation
    mid_idx = ceil(length(m)/2);
    alpha_mid = alpha_params(mid_idx);
    omega_mid = omega_params(mid_idx);
    gamma_mid = gamma_params(mid_idx);
    lambda_mid = lambda_params(mid_idx);
    
    % Generate stock prices and volatility using mcHN
    [S, h0] = mcHN(M, N, S_init, Z, r, omega_mid, alpha_mid, beta, gamma_mid, lambda_mid);

    % Determine stock price on the target day
    S0 = S(end + (-max_day + day_num), :);

    % Adjust maturities based on the day number
    % Given T = [5, 10, 21, 42, 63, 126, 180, 252, 360];
    % Replenish the maturity once it expires.
    t = T - (day_num - 1);
    % Generalized time cycling logic
    for i = 1:length(T)
        if t(i) <= 0
            t(i) = T(i) - mod(abs(t(i)), T(i));
        end
    end

    % Initialize dataset
    T_len = length(T);
    m_len = length(m);
    dataset = zeros(11, T_len * m_len * 2); % 2 for call and put options
    idx = 1; % Index counter for storing in the dataset

    % Generate dataset for each maturity and moneyness
    for i = 1:T_len
        for j = 1:m_len
            K = m(j) * S0; % Strike price
            
            % Use parameters specific to this moneyness
            alpha = alpha_params(j);
            omega = omega_params(j);
            gamma = gamma_params(j);
            lambda = lambda_params(j);

            % Calculate call and put option values with current parameters
            [V_C, ~] = datagen2(t(i), r, S0, h0, K, alpha, beta, omega, gamma, lambda, 1); % Call
            [V_P, ~] = datagen2(t(i), r, S0, h0, K, alpha, beta, omega, gamma, lambda, -1); % Put

            % Round option values to 4 decimal places
            V_C = round(V_C, 4);
            V_P = round(V_P, 4);

            % Store call option data
            dataset(:, idx) = [S0; K/S0; r; t(i); 1; alpha; beta; omega; gamma; lambda; V_C];
            idx = idx + 1;

            % Store put option data
            dataset(:, idx) = [S0; K/S0; r; t(i); -1; alpha; beta; omega; gamma; lambda; V_P];
            idx = idx + 1;
        end
    end
end