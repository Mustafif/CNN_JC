function [nodes_ht,p, hmom1, hmom2, hmom3, hmom4_app] = TreeNodes_ht_D(m_h, h0, gamma_h,beta_0, beta_1,beta_2, theta, lambda,N)
%
% compute the first four moments of h_t of the non-affine GARCH model proposed by Duan
% and construct all tree nodes of h_t under Q measure
%
%       X_t = r-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = beta_0+beta_1*h_t+beta_2*h_t(z_t-theat-lambad)^2
%
%
%  Input
%     m_h  -- number of tree nodes of h_t
%     h0 -- initial value
%     gamma_h -- parameter for zq
%     beta_0, beta_1, beta_2, theta, lambda -- parameters for Duan's GARCH
%     N -- # of time steps.
%
%  Output
%    modes_ht -- tree nodes for h_t
%    p -- probability of h_t at t=1 given h0
%    hmom1, hmom2, hmom3, hmom4 -- first four moments of ht 
%
%       April 26, 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu_1 = beta_2*(1+(lambda+theta)^2)+beta_1;
mu_2 = beta_2^2*(3+6*(lambda+theta)^2+(lambda+theta)^4) + ...
    2*beta_1*beta_2*(1+(lambda+theta)^2)+beta_1^2;
mu_3 = beta_2^3*(15+45*(lambda+theta)^2+15*(lambda+theta)^4+(lambda+theta)^6) + ...
    3*beta_1*beta_2^2*(3+6*(lambda+theta)^2+(lambda+theta)^4) + ...
    3*beta_1^2*beta_2*(1+(lambda+theta)^2) +beta_1^3;

% first four moments

n = 1:1:N+1;
n = n(:);
hmom1 = beta_0*(1-mu_1.^(n-1))./(1-mu_1)+mu_1.^(n-1).*h0;
hmom2 = beta_0^2.*((1+mu_1)./(1-mu_1).*(1-mu_2.^(n-1))./(1-mu_2) - 2*mu_1./(1-mu_1).*(mu_1.^(n-1)-mu_2.^(n-1))./(mu_1-mu_2)) + ...
       2*beta_0*mu_1.*(mu_1.^(n-1)-mu_2.^(n-1))./(mu_1-mu_2).*h0 + ...
       mu_2.^(n-1).*h0.^2;
hmom3 = beta_0^3.*((1-mu_3.^(n-1))./(1-mu_3)+3*(mu_2./(1-mu_2).*(1-mu_3.^(n-1))./(1-mu_3)-mu_2./(1-mu_2).*(mu_2.^(n-1)-mu_3.^(n-1))./(mu_2-mu_3)) +...
    3.*(mu_1./(1-mu_1).*(1-mu_3.^(n-1))./(1-mu_3)-mu_1./(1-mu_1).*(mu_1.^(n-1)-mu_3.^(n-1))./(mu_1-mu_3)) + ...
    6.*(mu_1./(1-mu_1).*mu_2./(1-mu_2).*(1-mu_3.^(n-1))./(1-mu_3)- mu_1./(1-mu_1).*mu_2./(1-mu_2).*(mu_2.^(n-1)-mu_3.^(n-1))./(mu_2-mu_3) - ...
    mu_1./(1-mu_1).*mu_2./(1-mu_2).*(mu_1.^(n-1)-mu_3.^(n-1))./(mu_1-mu_3) + mu_1./(1-mu_1).*mu_2./(1-mu_2).*(mu_2.^(n-1)-mu_3.^(n-1))./(mu_2-mu_3))) + ...
    3.*beta_0^2*mu_1.*h0.*((mu_1.^(n-1)-mu_3.^(n-1))./(mu_1-mu_3)+2.*(mu_2./(mu_1-mu_2).*(mu_1.^(n-1)-mu_3.^(n-1))./(mu_1-mu_3)- ...
    mu_2./(mu_1-mu_2).*(mu_2.^(n-1)-mu_3.^(n-1))./(mu_2-mu_3))) +...
    3*beta_0*mu_2*(mu_2.^(n-1)-mu_3.^(n-1))./(mu_2-mu_3).*h0^2 + mu_3.^(n-1)*h0.^3;

hmom3_app =3.*hmom1.*hmom2 - 2.*hmom1.^3;
% hmom3 = hmom3_app;
hmom4_app = 6.*hmom1.^2.*hmom2-5.*hmom1.^4;


%% generate nodes by 4th order moments
mu = hmom1(2:end);
var = hmom2(2:end)-mu.^2;
temp = sqrt(var);
k3 = 1./(temp.^3).*(hmom3(2:end)-3*mu.*hmom2(2:end)+2*mu.^3);
k4 = 1./(temp.^4).*(hmom4_app(2:end)-4*mu.*hmom3(2:end)+6*(mu.^2).*hmom2(2:end)-3*mu.^4);


G= [mu, var,k3,k4]';
[nodes_ht, p] = Treenodes_JC_X(G,N,m_h,gamma_h);

%p = Prob(h0, nodes_ht(:,1), beta_0,beta_1, beta_2, theta, lambda);
p = p(:);
end




