function ds = dataset_contract(contract)
    % Use original parameters
    alpha = 1.33e-6;
    beta = 0.8;
    omega = 1e-6;
    gamma = 100;
    lambda = 0.5;
    
    days_per_contract = 31;
    days = 1:days_per_contract;
    r = contract(1);
    T = contract(2:10);
    m = contract(11:19);
    
    % Define artificial volatility smile parameters for post-processing
    smile_center = 1.0;  % ATM moneyness
    smile_width = 0.15;  % How wide the smile is
    smile_depth = 0.05;  % How pronounced the smile is
    skew = 0.02;         % Skew parameter (asymmetry)
    
    ds = arrayfun(@(day) GenDaysData(days_per_contract, day, r, T, m, alpha, beta, omega, gamma, lambda, smile_center, smile_width, smile_depth, skew), days, 'UniformOutput', false);
    ds = cell2mat(ds);
end

function dataset = GenDaysData(max_day, day_num, r, T, m, alpha, beta, omega, gamma, lambda, smile_center, smile_width, smile_depth, skew)
    % Generate dataset for a given day number using market and GARCH parameters.
    % Inputs:
    % - day_num: Current day number
    % - r: Risk-free rate
    % - T: Maturity time points
    % - m: Moneyness factors
    % - alpha, beta, omega, gamma, lambda: GARCH model parameters
    % - smile parameters: to artificially create a volatility smile
    % Outputs:
    % - dataset: Generated dataset containing option pricing details

    % Simulation parameters
    N = 504; % Total trading days (2 years)
    M = 1;   % Number of Monte Carlo paths
    S_init = 100; % Initial stock price

    % Generate random standard normal samples
    Z = randn(N + 1, M);

    % Generate stock prices and volatility using mcHN
    [S, h0] = mcHN(M, N, S_init, Z, r, omega, alpha, beta, gamma, lambda);

    % Determine stock price on the target day
    S0 = S(end + (-max_day + day_num), :);

    % Adjust maturities based on the day number
    % Given T = [5, 10, 21, 42, 63, 126, 180, 252, 360];
    % Replenish the maturity once it expires.
    t = T - (day_num - 1);
    % Generalized time cycling logic
    for i = 1:length(T)
        if t(i) <= 0
            % Calculate how many full cycles have expired
            %expired_cycles = floor(abs(t(i)) / T(i)) + 1;
            
            % Restore maturity by applying the correct number of cycles
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
        % Time factor for maturity effect on volatility smile
        time_factor = sqrt(252/max(5, t(i)));
        
        for j = 1:m_len
            K = m(j) * S0; % Strike price
            
            % Get base option pricing
            [V_C, ~] = datagen2(t(i), r, S0, h0, K, alpha, beta, omega, gamma, lambda, 1); % Call
            [V_P, ~] = datagen2(t(i), r, S0, h0, K, alpha, beta, omega, gamma, lambda, -1); % Put
            
            % Apply artificial smile effect by manipulating option prices
            % This will result in implied vols with a smile pattern
            smile_effect = smile_depth * time_factor * ((m(j) - smile_center)/smile_width)^2;
            skew_effect = skew * time_factor * (m(j) - smile_center);
            
            % Add asymmetry
            total_effect = smile_effect + skew_effect;
            
            % Scale effect based on option value to maintain reasonable adjustments
            V_C_adj = V_C * (1 + total_effect);
            V_P_adj = V_P * (1 + total_effect);

            % Round option values to 4 decimal places
            V_C = round(V_C_adj, 4);
            V_P = round(V_P_adj, 4);

            % Store call option data
            dataset(:, idx) = [S0; K/S0; r; t(i); 1; alpha; beta; omega; gamma; lambda; V_C];
            idx = idx + 1;

            % Store put option data
            dataset(:, idx) = [S0; K/S0; r; t(i); -1; alpha; beta; omega; gamma; lambda; V_P];
            idx = idx + 1;
        end
    end
end