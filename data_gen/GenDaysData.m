function dataset = GenDaysData(max_day, day_num, r, T, m, alpha, beta, omega, gamma, lambda)
    % Generate dataset for a given day number using market and GARCH parameters.
    % Inputs:
    % - day_num: Current day number
    % - r: Risk-free rate
    % - T: Maturity time points
    % - m: Moneyness factors
    % - alpha, beta, omega, gamma, lambda: GARCH model parameters
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
        for j = 1:m_len
            K = m(j) * S0; % Strike price

            % Calculate call and put option values
            [V_C, ~] = datagen2(t(i), r, S0, h0, K, alpha, beta, omega, gamma, lambda, 1); % Call
            [V_P, ~] = datagen2(t(i), r, K, h0, S0, alpha, beta, omega, gamma, lambda, -1); % Put

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
