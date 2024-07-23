function [nodes_ht, mu,var, k3,k4] = TreeNodes_ht_HN_New(m_h, hd, qht, gamma_h,alpha,beta,gamma,omega,N)
%
% compute the first four moments of h_t of the Heston-Nadi GARCH model and
% construct all tree nodes of h_t
%
%       X_t = r-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega +alpha(z_t-gamma*sqrt(h_t))^2+beta h_t
%
%
%  Input
%     m_h  -- number of tree nodes of h_t
%     hd -- discrete values of h_1
%     qht -- corresponding probabilities of hd
%     gamma_h -- parameter for generating z_t
%     alpha, beta, gamma, omega -- parameters for Heston-Nadi GARCH
%     N -- # of time steps.
%
%       April 22, 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numPoint = N+1;
u =0;
%% Recursivesly compute the derivatives of B at u=0
% 0th order

diffB0 = zeros(numPoint,1);
diffB0(1) = 0; % for u=0

% diffB0(1) = 0; 
% for row = 2:numPoint
% diffB0(row) = -0.5*gamma^2 + (-0.5 + gamma)*u + beta*diffB0(row-1) + (-gamma + u)^2/(2*(1 - 2*alpha*diffB0(row-1)));
% end

% 1st order
for row = 2:numPoint
    diffB0(row) = beta*diffB0(row-1)+(alpha*gamma^2*diffB0(row-1))/(1-2*alpha*diffB0(row-1));
end

% 1st order
diffB1 = zeros(numPoint,1);
diffB1(1) = 1; % 1st order derivative equals 1
for row = 2:numPoint
    diffB1(row) = ((beta + alpha*gamma^2 + 4*alpha*beta*diffB0(row-1)*(-1 + alpha*diffB0(row-1)))*diffB1(row-1))/(1 - 2*alpha*diffB0(row-1))^2;
end

% 2nd order
diffB2 = zeros(numPoint,1);
diffB2(1) = 0; % 2nd order derivative equals 0
for row = 2:numPoint
    diffB2(row) = beta*diffB2(row-1) + (8*alpha^3*gamma^2*diffB0(row-1)*diffB1(row-1).^2)/(1 - 2*alpha*diffB0(row-1)).^3+ ...
        (2*alpha^2*gamma^2*(2*diffB1(row-1).^2+diffB0(row-1)*diffB2(row-1)))/(1 - 2*alpha*diffB0(row-1)).^2 + ...
        alpha*gamma^2*diffB2(row-1)/(1 - 2*alpha*diffB0(row-1));
end

% 3rd order
diffB3 = zeros(numPoint,1);
diffB3(1) = 0; %3rd order derivative equals 0
for row = 2:numPoint
    diffB3(row) = beta*diffB3(row-1) + 48*alpha^4*gamma^2*diffB0(row-1)*diffB1(row-1).^3/(1 - 2*alpha*diffB0(row-1))^4 + ...
        + (24*alpha^3*gamma^2*diffB1(row-1).^3+24*alpha^3*gamma^2*diffB0(row-1)*diffB1(row-1)*diffB2(row-1))/(1 - 2*alpha*diffB0(row-1))^3 + ...
        +(12*alpha^2*gamma^2*diffB1(row-1)*diffB2(row-1) +2*alpha^2*gamma^2*diffB0(row-1)*diffB3(row-1))/(1 - 2*alpha*diffB0(row-1))^2 + ...
        alpha*gamma^2*diffB3(row-1)/(1 - 2*alpha*diffB0(row-1));
end

% 4th order
diffB4 = zeros(numPoint,1);
diffB4(1) = 0; %4th order derivative equals 0
for row = 2:numPoint
    
    diffB4(row) = beta*diffB4(row-1) + 384*alpha^5*gamma^2*diffB0(row-1)*diffB1(row-1)^4/(1 - 2*alpha*diffB0(row-1))^5 + ...
        (192*alpha^4*gamma^2*diffB1(row-1)^4+288*alpha^4*gamma^2*diffB0(row-1)*diffB1(row-1)^2*diffB2(row-1))/(1 - 2*alpha*diffB0(row-1))^4 + ...
        (144*alpha^3*gamma^2*diffB1(row-1)^2*diffB2(row-1)+24*alpha^3*gamma^2*diffB0(row-1)*diffB2(row-1)^2+32*alpha^3*gamma^2*diffB0(row-1)*diffB1(row-1)*diffB3(row-1))/(1 - 2*alpha*diffB0(row-1))^3 +...
        (12*alpha^2*gamma^2*diffB2(row-1)^2+16*alpha^2*gamma^2*diffB1(row-1)*diffB3(row-1)+2*alpha^2*gamma^2*diffB0(row-1)*diffB4(row-1))/(1 - 2*alpha*diffB0(row-1))^2 +...
        alpha*gamma^2*diffB4(row-1)/(1 - 2*alpha*diffB0(row-1));
end

%% Recursively compute the derivatives of A at u=0
% 0th order
diffA0 = zeros(numPoint,1);
diffA0(1) = 0;
for row = 2:numPoint
    diffA0(row) =  diffA0(row-1)+omega*diffB0(row-1)-1/2*log(1-2*alpha*diffB0(row-1));
end

% 1st order
diffA1 = zeros(numPoint,1);
diffA1(1) = 0;
for row = 2:numPoint
    diffA1(row) =  diffA1(row-1) + (omega + (1.*alpha)/(1 - 2*alpha*diffB0(row-1)))*diffB1(row-1);
end

% 2nd order
diffA2 = zeros(numPoint,1);
diffA2(1) = 0;
for row = 2:numPoint
    diffA2(row) =  (2.*alpha^2*diffB1(row-1)^2)/(1 - 2*alpha*diffB0(row-1))^2 + diffA2(row-1) + ...
        (omega + (1.*alpha)/(1 - 2*alpha*diffB0(row-1)))*diffB2(row-1);
end

% 3rd order
diffA3 = zeros(numPoint,1);
diffA3(1) = 0;
for row = 2:numPoint
    diffA3(row) =  (8.*alpha^3*diffB1(row-1)^3)/(1 - 2*alpha*diffB0(row-1))^3 +...
        (6.*alpha^2*diffB1(row-1)*diffB2(row-1))/(1 - 2*alpha*diffB0(row-1))^2 + ...
        diffA3(row-1) + (omega + (1.*alpha)/(1 - 2*alpha*diffB0(row-1)))*diffB3(row-1);
end

% 4th order
diffA4 = zeros(numPoint,1);
diffA4(1) = 0;
for row = 2:numPoint
diffA4(row) = diffA4(row-1) + omega*diffB4(row-1)+(48*alpha^4*diffB1(row-1).^4)/(1-2*alpha*diffB0(row-1)).^4 ...
    +(48*alpha.^3*diffB1(row-1).^2.*diffB2(row-1))/(1-2*alpha.*diffB0(row-1)).^3 + ...
    (6*alpha^2*diffB2(row-1).^2+8*alpha.^2.*diffB1(row-1).*diffB3(row-1))/(1-2*alpha.*diffB0(row-1)).^2 +...
    alpha.*diffB4(row-1)/(1-2*alpha.*diffB0(row-1));
end

%%Recursively compute the derivatives of m.g.f at u=0

tmp1 = ones(m_h,1).*diffA0' + hd.*diffB0';
tmp2 = ones(m_h,1).*diffA1' + hd.*diffB1';
tmp3 = ones(m_h,1).*diffA2' + hd.*diffB2';
tmp4 = ones(m_h,1).*diffA3' + hd.*diffB3';
% 0th order
% diffmgf0 = exp(diffA0+hd.*diffB0);

% 1st order
diffmgf1 = exp(tmp1).*tmp2;

% 2nd order
diffmgf2 = exp(tmp1).*((tmp2).^2 + tmp3) ;

% 3rd order
diffmgf3 = exp(tmp1).*(tmp2.^3 + 3.*(tmp2).*(tmp3) + tmp4);

% 4th order
diffmgf4 = exp(tmp1).*((tmp2).^4 + ...
    6.*(tmp2).^2.*(tmp3) + 3.*(tmp3).^2 + ...
    4.*(tmp2).*(tmp4) + ones(m_h,1).*diffA4' + hd.*diffB4');

mom1 = qht'*diffmgf1(:,2:end);
mom2 = qht'*diffmgf2(:,2:end);
mom3 = qht'*diffmgf3(:,2:end);
mom4 = qht'*diffmgf4(:,2:end);

%% generate nodes by 4th order moments
mu = mom1;
var = mom2-mu.^2;
temp = sqrt(var);
k3 = 1./(temp.^3).*(mom3-3*mu.*mom2+2*mu.^3);
k4 = 1./(temp.^4).*(mom4-4*mu.*mom3+6*(mu.^2).*mom2-3*mu.^4);

G= [mu;var;k3;k4];
[nodes_ht] = Treenodes_JC_h(G,N,m_h,gamma_h);

end




