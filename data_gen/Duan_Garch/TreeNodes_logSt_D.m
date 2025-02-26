function [nodes_Xt,mu,var,k3, k4] = TreeNodes_logSt_D(m_x,gamma_x,r, h0, beta_0,beta_1, beta_2,theta, lambda,N, hmom1, hmom2)
%
% compute the first four moments and tree nodes of X_t of the Duan's GARCH model and
% construct all tree nodes of X_t
%
%       X_t = r-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = beta_0+beta_1*h_t+beta_2*h_t*(z_t-theta-lambda)^2
%
%
%  Input
%     m_x  -- number of tree nodes
%     gamma_x -- parameter for generating z_t
%     r -- interest rate
%     hd -- discrete values of h_1
%     qht -- corresponding probabilities of hd
%     S0 -- intial stock price
%     alpha, beta, gamma, omega -- parameters for Heston-Nadi GARCH
%     N -- # of time steps.
%
%       April 22, 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% second moment
mu_1 = beta_2*(1+(lambda+theta)^2)+beta_1;
Ymom1 = zeros(N,1);
Ymom2 = zeros(N,1);
for n = 1:N
    % first moment
    Ymom1(n) = n*r - 0.5*sum(hmom1(1:n)); % Xt = ln(St/S0)
    
    tmp = sum(hmom2(1:n));
    for i = 1:n
        for j = 1: n-i
            tmp = tmp + 2*(beta_0*(1-mu_1^j)/(1-mu_1)*hmom1(i)+mu_1^j*hmom2(i));
        end
    end
    SD1 = tmp;
    SD2 = sum(hmom1(1:n));
    
    v_1 = -2*beta_2*(lambda+theta);
    SD3 = 0;
    for i = 1:n
        for j = 1:n-i
            SD3 = SD3 + v_1*mu_1^(j-1)*(3/8*hmom1(i)^(-0.5)*hmom2(i)+5/8*hmom2(i)^(1.5));
        end
    end
    
    Ymom2(n) = n^2*r^2-n*r*sum(hmom1(1:n))+0.25*SD1+SD2-SD3;
    
end
% Approximate third moment
tmp = (r-0.5*h0)^3+3*(r-0.5*h0)*h0;
Ymom3 = tmp*ones(N,1);
% Approximate fourth moment
tmp = (r-0.5*h0)^4+6*(r-0.5*h0)^4*h0+3*h0^2;
Ymom4 = tmp*ones(N,1);



%%  generate nodes by the fourth order moments
mu = Ymom1;
var = (Ymom2-mu.^2);
temp = sqrt(var);
k3 = 1./(temp(1).^3).*(Ymom3-3.*mu(1).*Ymom2(1)+2.*mu(1).^3);
k4 = 1./(temp(1).^4).*(Ymom4-4.*mu(1).*Ymom3+6.*(mu(1).^2).*Ymom2(1)-3.*mu(1).^4);
% tmp3 = k3(:,1);
% tmp4 = k4(:,1);
% k33 = repmat(tmp3, 1,N);
% k44 = repmat(tmp4, 1, N);

G = [mu, var, k3, k4]';
nodes_Xt = Treenodes_JC_X(G,N,m_x,gamma_x);

end