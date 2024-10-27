function [nodes_Xt, mu, var, k3, k4] = TreeNodes_logSt_Duan(m_x, r, hd, qht, S0, alpha, beta, gamma, omega, N)
% Compute the first four moments of X_t of the Heston-Nandi GARCH model and
% construct tree nodes using Duan's method
%
% Input:
%     m_x  -- number of tree nodes
%     r -- interest rate
%     hd -- discrete values of h_1
%     qht -- corresponding probabilities of hd
%     S0 -- initial stock price
%     alpha, beta, gamma, omega -- parameters for Heston-Nandi GARCH
%     N -- # of time steps
%
% Output:
%     nodes_Xt -- tree nodes for X_t
%     mu -- mean
%     var -- variance
%     k3 -- third moment (skewness)
%     k4 -- fourth moment (kurtosis)

% Initialize arrays for derivatives
numPoint = N + 1;
[diffB0, diffB1, diffB2, diffB3, diffB4] = computeDiffB(numPoint, alpha, beta, gamma);
[diffA0, diffA1, diffA2, diffA3, diffA4] = computeDiffA(numPoint, r, omega, alpha, diffB0, diffB1, diffB2, diffB3, diffB4);

% Compute moments
[mu, var, k3, k4] = computeMoments(diffA0, diffA1, diffA2, diffA3, diffA4, ...
                                  diffB0, diffB1, diffB2, diffB3, diffB4, ...
                                  hd, qht, S0, N);

% Generate nodes using Duan's method
nodes_Xt = generateDuanNodes(mu, var, k3, k4, m_x, N);
end

function [diffB0, diffB1, diffB2, diffB3, diffB4] = computeDiffB(numPoint, alpha, beta, gamma)
    % Initialize arrays
    diffB0 = zeros(numPoint, 1);
    diffB1 = zeros(numPoint, 1);
    diffB2 = zeros(numPoint, 1);
    diffB3 = zeros(numPoint, 1);
    diffB4 = zeros(numPoint, 1);
    
    % Compute derivatives recursively
    for row = 2:numPoint
        % B0 derivative
        diffB0(row) = beta * diffB0(row-1) + ...
                      (alpha * gamma^2 * diffB0(row-1)) / (1 - 2*alpha*diffB0(row-1));
        
        % B1 derivative
        diffB1(row) = -0.5 + beta * diffB1(row-1) + ...
                      (alpha * gamma^2 * diffB1(row-1)) / (1 - 2*alpha*diffB0(row-1)) + ...
                      (2 * alpha^2 * gamma^2 * diffB0(row-1) * diffB1(row-1)) / (1 - 2*alpha*diffB0(row-1))^2;
        
        % B2 derivative
        diffB2(row) = beta * diffB2(row-1) + ...
                      (alpha * gamma^2 * diffB2(row-1)) / (1 - 2*alpha*diffB0(row-1)) + ...
                      (4 * alpha^2 * gamma^2 * diffB1(row-1)^2) / (1 - 2*alpha*diffB0(row-1))^2;
        
        % B3 derivative
        diffB3(row) = beta * diffB3(row-1) + ...
                      (alpha * gamma^2 * diffB3(row-1)) / (1 - 2*alpha*diffB0(row-1)) + ...
                      (12 * alpha^2 * gamma^2 * diffB1(row-1) * diffB2(row-1)) / (1 - 2*alpha*diffB0(row-1))^2;
        
        % B4 derivative
        diffB4(row) = beta * diffB4(row-1) + ...
                      (alpha * gamma^2 * diffB4(row-1)) / (1 - 2*alpha*diffB0(row-1)) + ...
                      (12 * alpha^2 * gamma^2 * diffB2(row-1)^2 + ...
                       24 * alpha^2 * gamma^2 * diffB1(row-1) * diffB3(row-1)) / (1 - 2*alpha*diffB0(row-1))^2;
    end
end

function [diffA0, diffA1, diffA2, diffA3, diffA4] = computeDiffA(numPoint, r, omega, alpha, diffB0, diffB1, diffB2, diffB3, diffB4)
    % Initialize arrays
    diffA0 = zeros(numPoint, 1);
    diffA1 = zeros(numPoint, 1);
    diffA2 = zeros(numPoint, 1);
    diffA3 = zeros(numPoint, 1);
    diffA4 = zeros(numPoint, 1);
    
    % Compute derivatives recursively
    for row = 2:numPoint
        % A0 derivative
        diffA0(row) = diffA0(row-1) + r + diffB0(row-1) * omega - ...
                      0.5 * log(1 - 2 * diffB0(row-1) * alpha);
        
        % A1 derivative
        diffA1(row) = r + diffA1(row-1) + ...
                      (omega + alpha / (1 - 2*alpha*diffB0(row-1))) * diffB1(row-1);
        
        % A2 derivative
        diffA2(row) = diffA2(row-1) + ...
                      (2 * alpha^2 * diffB1(row-1)^2) / (1 - 2*alpha*diffB0(row-1))^2 + ...
                      (omega + alpha / (1 - 2*alpha*diffB0(row-1))) * diffB2(row-1);
        
        % A3 derivative
        diffA3(row) = diffA3(row-1) + ...
                      (6 * alpha^2 * diffB1(row-1) * diffB2(row-1)) / (1 - 2*alpha*diffB0(row-1))^2 + ...
                      (omega + alpha / (1 - 2*alpha*diffB0(row-1))) * diffB3(row-1);
        
        % A4 derivative
        diffA4(row) = diffA4(row-1) + omega * diffB4(row-1) + ...
                      (6 * alpha^2 * diffB2(row-1)^2 + 8 * alpha^2 * diffB1(row-1) * diffB3(row-1)) / ...
                      (1 - 2*alpha*diffB0(row-1))^2;
    end
end

function [mu, var, k3, k4] = computeMoments(diffA0, diffA1, diffA2, diffA3, diffA4, ...
                                          diffB0, diffB1, diffB2, diffB3, diffB4, ...
                                          hd, qht, S0, N)
    % Compute temporary matrices for moment calculations
    m_h = length(hd);
    tmp1 = ones(m_h,1) .* diffA0' + hd .* diffB0';
    tmp2 = ones(m_h,1) .* diffA1' + hd .* diffB1';
    tmp3 = ones(m_h,1) .* diffA2' + hd .* diffB2';
    tmp4 = ones(m_h,1) .* diffA3' + hd .* diffB3';
    
    % Compute MGF derivatives
    diffmgf1 = exp(tmp1) .* (log(S0) + tmp2);
    diffmgf2 = exp(tmp1) .* (log(S0).^2 + 2*tmp2.*log(S0) + tmp2.^2 + tmp3);
    diffmgf3 = exp(tmp1) .* ((log(S0) + tmp2).^3 + 3*(log(S0) + tmp2).*tmp3 + tmp4);
    diffmgf4 = exp(tmp1) .* ((log(S0) + tmp2).^4 + 6*(log(S0) + tmp2).^2.*tmp3 + ...
               3*tmp3.^2 + 4*(log(S0) + tmp2).*tmp4 + ones(m_h,1).*diffA4' + hd.*diffB4');
    
    % Compute moments
    mu = qht' * diffmgf1(:,2:end);
    var = qht' * diffmgf2(:,2:end) - mu.^2;
    temp = sqrt(var);
    k3 = (qht' * diffmgf3(:,2:end) - 3*mu.*var - mu.^3) ./ temp.^3;
    k4 = (qht' * diffmgf4(:,2:end) - 4*mu.*qht'*diffmgf3(:,2:end) + ...
          6*(mu.^2).*var + 3*mu.^4) ./ var.^2;
    
    % Replicate moments for all time steps
    k3 = repmat(k3(1), 1, N);
    k4 = repmat(k4(1), 1, N);
end

function nodes = generateDuanNodes(mu, var, k3, k4, m_x, N)
    % Generate tree nodes using Duan's method
    nodes = zeros(m_x, N);
    
    for t = 1:N
        % Compute standardized nodes using Duan's method
        std_nodes = getDuanPoints(m_x, k3(t), k4(t));
        
        % Transform standardized nodes to actual nodes
        nodes(:,t) = mu(t) + sqrt(var(t)) * std_nodes;
    end
end

function points = getDuanPoints(n, skewness, kurtosis)
    % Implementation of Duan's method for generating points
    % that match the first four moments
    
    % Initialize optimization parameters
    options = optimset('Display', 'off', 'MaxFunEvals', 1000);
    x0 = zeros(n-1, 1);  % Initial guess for node locations
    
    % Objective function for moment matching
    objFun = @(x) momentMatchingError(x, n, skewness, kurtosis);
    
    % Solve optimization problem
    [x_opt,~] = fminsearch(objFun, x0, options);
    
    % Generate final points
    points = [-flip(x_opt); 0; x_opt];
end

function err = momentMatchingError(x, n, target_skew, target_kurt)
    % Compute error between target and achieved moments
    points = [-flip(x); 0; x];
    probs = ones(2*n-1, 1) / (2*n-1);
    
    % Compute actual moments
    m1 = sum(points .* probs);
    m2 = sum((points - m1).^2 .* probs);
    m3 = sum((points - m1).^3 .* probs) / m2^1.5;
    m4 = sum((points - m1).^4 .* probs) / m2^2;
    
    % Compute error
    err = (m3 - target_skew)^2 + (m4 - target_kurt)^2;
end