function price = priceAmericanDuanGARCH(S0, K, T, r, alpha, beta, omega, gamma, lambda, N)
    % Parameters:
    % S0 - Initial stock price
    % K - Strike price
    % T - Time to maturity (in days)
    % r - Risk-free rate
    % alpha, beta, omega, gamma, lambda - GARCH(1,1) parameters
    % N - Number of time steps in the tree

    dt = T / N; % Time step
    
    % Compute initial variance from stationary condition
    h0 = omega / (1 - alpha - beta);
    
    % Generate the Willow tree structure
    S = zeros(N+1, N+1);
    h = zeros(N+1, N+1);
    P = zeros(N, N+1, N+1); % Transition probabilities
    
    S(1,1) = S0;
    h(1,1) = h0;
    
    for i = 1:N
        for j = 1:i
            z = norminv((j-0.5)/(i+1)); % Avoid out-of-bounds probabilities
            h(i+1, j) = max(omega + beta * h(i, j) + alpha * (sqrt(h(i, j)) * z - gamma)^2, 1e-8);
            S(i+1, j) = S(i, j) * exp(r * dt - 0.5 * h(i+1, j) * dt + sqrt(h(i+1, j) * dt) * z);
            
            % Ensure valid transition probabilities
            if j < i
                P(i, j, j) = 0.5;
                P(i, j, j+1) = 0.5;
            end
        end
    end
    
    % Backward induction for American option pricing
    V = max(S - K, 0);
    for i = N:-1:1
        for j = 1:i
            continuation = exp(-r * dt) * (P(i, j, j) * V(i+1, j) + P(i, j, j+1) * V(i+1, j+1));
            V(i, j) = max(continuation, max(S(i, j) - K, 0));
        end
    end
    
    price = V(1,1);
end