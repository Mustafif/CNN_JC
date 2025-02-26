function params = genHNGarchParams(n)
    % Initialize an array to store the parameters
    params = zeros(n, 5);

    % Define the ranges for each parameter
    omegaRange = [1e-7, 1e-6]; % Around 1e-6
    alphaRange = [1.15e-6, 1.36e-6]; % Around 1.33e-6
    betaRange = [0.75, 0.85]; % Around 0.8
    gammaRange = [50, 150]; % Around 100
    lambdaRange = [0.4, 0.6]; % Around 0.5

    % Generate valid parameters satisfying the stationarity condition
    for i = 1:n
        valid = false;
        while ~valid
            omega = omegaRange(1) + (omegaRange(2) - omegaRange(1)) * rand();
            alpha = alphaRange(1) + (alphaRange(2) - alphaRange(1)) * rand();
            beta = betaRange(1) + (betaRange(2) - betaRange(1)) * rand();
            gamma = gammaRange(1) + (gammaRange(2) - gammaRange(1)) * rand();
            lambda = lambdaRange(1) + (lambdaRange(2) - lambdaRange(1)) * rand();
            
            % Ensure stationarity: beta + alpha * gamma^2 < 1
            % Additionally, alpha + beta < 1 for covariance stationarity
            if (beta + alpha * gamma^2 < 1) && (alpha + beta < 1)
                params(i, :) = [alpha, beta, omega, gamma, lambda];
                valid = true;
            end
        end
    end
end
