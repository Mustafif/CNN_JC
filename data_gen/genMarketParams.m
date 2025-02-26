function params = genMarketParams(n)
    % Market Parameters:
    % Each row represents [m, r, T]
    params = zeros(n, 19); 

    % Define moneyness and time to maturity ranges
    mRange = [0.8, 1.2];  % Moneyness range
    TRange = [5, 360];    % Time to maturity range (in days)

    % Fixed risk-free rate
    r = 0.03;

    for i = 1:n
        % Randomly generate moneyness in [0.8, 1.2]
        m = mRange(1) + (mRange(2) - mRange(1)) * rand(1,9);

        % Randomly generate T in [5, 360]
        T = TRange(1) + (TRange(2) - TRange(1)) * rand(1,9);

        % Store in params matrix
        params(i, :) = [m, r, T];
    end
end
