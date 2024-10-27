function [nodes_ht, mu, var, k3, k4] = TreeNodes_ht_Duan(m_h, hd, qht, gamma_h, alpha, beta, omega, N)
% Compute the first four moments of h_t of Duan's GARCH model and
% construct all tree nodes of h_t
%
% Duan's GARCH specification:
%       X_t = r + lambda*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega + alpha*(r_t - r - lambda*h_t)^2 + beta*h_t
%
% Input:
%     m_h  -- number of tree nodes of h_t
%     hd -- discrete values of h_1
%     qht -- corresponding probabilities of hd
%     gamma_h -- parameter for generating z_t
%     alpha, beta, omega -- GARCH parameters
%     N -- # of time steps

numPoint = N + 1;

% Initialize arrays for moment derivatives
diffB0 = zeros(numPoint, 1);
diffB1 = zeros(numPoint, 1);
diffB2 = zeros(numPoint, 1);
diffB3 = zeros(numPoint, 1);
diffB4 = zeros(numPoint, 1);

% Initialize first values
diffB0(1) = 0;
diffB1(1) = 1;
diffB2(1) = 0;
diffB3(1) = 0;
diffB4(1) = 0;

% Compute B derivatives under Duan's specification
% check this 
for row = 2:numPoint
    % 0th order
    diffB0(row) = beta * diffB0(row-1) + alpha * diffB0(row-1)/(1 - 2*alpha*diffB0(row-1));
    
    % 1st order
    diffB1(row) = beta * diffB1(row-1) + (alpha * (1 + 2*diffB0(row-1)) * diffB1(row-1))/(1 - 2*alpha*diffB0(row-1))^2;
    
    % 2nd order
    diffB2(row) = beta * diffB2(row-1) + ...
        (2*alpha * diffB1(row-1)^2)/(1 - 2*alpha*diffB0(row-1))^2 + ...
        (alpha * (1 + 2*diffB0(row-1)) * diffB2(row-1))/(1 - 2*alpha*diffB0(row-1))^2;
    
    % 3rd order
    diffB3(row) = beta * diffB3(row-1) + ...
        (6*alpha * diffB1(row-1) * diffB2(row-1))/(1 - 2*alpha*diffB0(row-1))^2 +...
        (alpha * (1 + 2*diffB0(row-1)) * diffB3(row-1))/(1 - 2*alpha*diffB0(row-1))^2;
    
    % 4th order
    diffB4(row) = beta * diffB4(row-1) + ...
        (8*alpha * diffB1(row-1) * diffB3(row-1) + 6*alpha * diffB2(row-1)^2)/(1 - 2*alpha*diffB0(row-1))^2 +...
        (alpha * (1 + 2*diffB0(row-1)) * diffB4(row-1))/(1 - 2*alpha*diffB0(row-1))^2;
end

% Initialize and compute A derivatives
diffA0 = zeros(numPoint, 1);
diffA1 = zeros(numPoint, 1);
diffA2 = zeros(numPoint, 1);
diffA3 = zeros(numPoint, 1);
diffA4 = zeros(numPoint, 1);

for row = 2:numPoint
    % 0th order
    diffA0(row) = diffA0(row-1) + omega * diffB0(row-1) - 0.5 * log(1 - 2*alpha*diffB0(row-1));
    
    % 1st order
    diffA1(row) = diffA1(row-1) + omega * diffB1(row-1) + alpha/(1 - 2*alpha*diffB0(row-1)) * diffB1(row-1);
    
    % 2nd order
    diffA2(row) = diffA2(row-1) + omega * diffB2(row-1) + ...
        (2*alpha^2 * diffB1(row-1)^2)/(1 - 2*alpha*diffB0(row-1))^2 +...
        alpha/(1 - 2*alpha*diffB0(row-1)) * diffB2(row-1);
    
    % 3rd order
    diffA3(row) = diffA3(row-1) + omega * diffB3(row-1) +...
        (6*alpha^2 * diffB1(row-1) * diffB2(row-1))/(1 - 2*alpha*diffB0(row-1))^2 +...
        alpha/(1 - 2*alpha*diffB0(row-1)) * diffB3(row-1);
    
    % 4th order
    diffA4(row) = diffA4(row-1) + omega * diffB4(row-1) +...
        (8*alpha^2 * diffB1(row-1) * diffB3(row-1) + 6*alpha^2 * diffB2(row-1)^2)/(1 - 2*alpha*diffB0(row-1))^2 +...
        alpha/(1 - 2*alpha*diffB0(row-1)) * diffB4(row-1);
end

% Compute moment generating function derivatives
tmp1 = ones(m_h,1) .* diffA0' + hd .* diffB0';
tmp2 = ones(m_h,1) .* diffA1' + hd .* diffB1';
tmp3 = ones(m_h,1) .* diffA2' + hd .* diffB2';
tmp4 = ones(m_h,1) .* diffA3' + hd .* diffB3';

% Compute MGF derivatives
diffmgf1 = exp(tmp1) .* tmp2;
diffmgf2 = exp(tmp1) .* (tmp2.^2 + tmp3);
diffmgf3 = exp(tmp1) .* (tmp2.^3 + 3.*tmp2.*tmp3 + tmp4);
diffmgf4 = exp(tmp1) .* (tmp2.^4 + 6.*tmp2.^2.*tmp3 + 3.*tmp3.^2 + ...
    4.*tmp2.*(ones(m_h,1).*diffA4' + hd.*diffB4'));

% Compute moments
mom1 = qht' * diffmgf1(:,2:end);
mom2 = qht' * diffmgf2(:,2:end);
mom3 = qht' * diffmgf3(:,2:end);
mom4 = qht' * diffmgf4(:,2:end);

% Compute standardized moments
mu = mom1;
var = mom2 - mu.^2;
temp = sqrt(var);
k3 = (mom3 - 3*mu.*mom2 + 2*mu.^3) ./ temp.^3;
k4 = (mom4 - 4*mu.*mom3 + 6*(mu.^2).*mom2 - 3*mu.^4) ./ temp.^4;

% Generate tree nodes
G = [mu; var; k3; k4];
[nodes_ht] = Treenodes_JC_h(G, N, m_h, gamma_h);

end