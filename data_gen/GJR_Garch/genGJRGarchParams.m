% function params = genGJRGarchParams(n)
%     % Initialize an array to store the parameters
%     params = zeros(n, 5);
% 
%     % Define the ranges for each parameter
%     omegaRange = [1e-7, 1e-6]; % Around 1e-6
%     alphaRange = [1.15e-6, 1.36e-6]; % Around 1.33e-6
%     betaRange = [0.75, 0.85]; % Around 0.8
%     % gammaRange = [50, 150]; % Around 100
%     lambdaRange = [0.4, 0.6]; % Around 0.5
% 
%     % Generate valid parameters satisfying the stationarity condition
%     for i = 1:n
%         valid = false;
%         while ~valid
%             omega = omegaRange(1) + (omegaRange(2) - omegaRange(1)) * rand();
%             alpha = alphaRange(1) + (alphaRange(2) - alphaRange(1)) * rand();
%             beta = betaRange(1) + (betaRange(2) - betaRange(1)) * rand();
%             % gamma = gammaRange(1) + (gammaRange(2) - gammaRange(1)) * rand();
%             lambda = lambdaRange(1) + (lambdaRange(2) - lambdaRange(1)) * rand();
% 
%             % Ensure stationarity: beta + alpha * gamma^2 < 1
%             % Additionally, alpha + beta < 1 for covariance stationarity
%             if (alpha + beta < 1)
%                 params(i, :) = [alpha, beta, omega, 0, lambda];
%                 valid = true;
%             end
%         end
%     end
% end

function params = genGJRGarchParams(n)
% genGJRGarchParams Generate valid parameter sets for GJR-GARCH(1,1) model
%`
% Input:
%   n      - Number of parameter sets to generate
%
% Output:
%   params - Matrix of size (n x 5) containing parameter sets
%           [omega, alpha, beta, lambda, h0]
%           where h0 is the initial long-run variance
%
% Note: Parameters are generated to ensure both stationarity and
% non-negativity conditions of the GJR-GARCH model

    % Input validation
    validateattributes(n, {'numeric'}, {'positive', 'integer', 'scalar'});
    
    % Initialize output matrix
    params = zeros(n, 5);
    
    % Define parameter ranges based on empirical studies
    % These ranges are typical for daily financial returns
    omegaRange = [1e-7, 1e-6];    % Constant term
    alphaRange = [0.01, 0.15];    % ARCH effect
    betaRange = [0.75, 0.85];     % GARCH persistence
    lambdaRange = [0.01, 0.15];   % Leverage effect
    
    % Maximum attempts to find valid parameters
    maxAttempts = 1000;
    
    % Generate parameter sets
    for i = 1:n
        attempts = 0;
        valid = false;
        
        while ~valid && attempts < maxAttempts
            % Generate random parameters
            omega = omegaRange(1) + (omegaRange(2) - omegaRange(1)) * rand();
            alpha = alphaRange(1) + (alphaRange(2) - alphaRange(1)) * rand();
            beta = betaRange(1) + (betaRange(2) - betaRange(1)) * rand();
            lambda = lambdaRange(1) + (lambdaRange(2) - lambdaRange(1)) * rand();
            
            % Check stationarity conditions for GJR-GARCH:
            % 1. beta + alpha + (lambda/2) < 1 (stationarity)
            % 2. omega > 0 (positive constant)
            % 3. alpha >= 0 (non-negative ARCH effect)
            % 4. beta >= 0 (non-negative GARCH effect)
            % 5. alpha + lambda >= 0 (non-negative news impact)
            
            if (beta + alpha + (lambda/2) < 1) && ...
               (omega > 0) && ...
               (alpha >= 0) && ...
               (beta >= 0) && ...
               (alpha + lambda >= 0)
                
                % Calculate long-run variance
                h0 = omega / (1 - beta - alpha - (lambda/2));
                
                % Store parameters: [omega, alpha, beta, lambda, h0]
                params(i, :) = [alpha, beta, omega, 0, lambda];
                valid = true;
            end
            
            attempts = attempts + 1;
        end
        
        if ~valid
            error('Failed to generate valid parameters after %d attempts for set %d', maxAttempts, i);
        end
    end
    
    % Verify all parameter sets are valid
    if any(params(:,1) <= 0) || ...  % omega > 0
       any(params(:,2) < 0) || ...   % alpha >= 0
       any(params(:,3) < 0) || ...   % beta >= 0
       any(params(:,2) + params(:,3) + params(:,4)/2 >= 1)  % stationarity condition
        error('Invalid parameter set generated. Please check the constraints.');
    end
end