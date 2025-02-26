function [nodes_Xt,mu_X,var,k3, k4] = TreeNodes_logSt_GJR(m_x,m_h,gamma_x,mu,w, alpha, lambda, beta,h0, N, hmom1, hmom2)
%
% compute the first four moments and tree nodes of X_t of the GJR GARCH model and
% construct all tree nodes of X_t
%
%       X_t = mu + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = w+beta*h_t+h_t(alpha+lambda*I_{z_t<0})z_t^2
%
%
%  Input
%     m_x  -- number of tree nodes
%     m_h -- number of ht
%     gamma_x -- parameter for generating z_t
%     hd -- discrete values of h_1
%     qht -- corresponding probabilities of hd
%     h0 -- intial value of variance
%     mu,w,beta,alpha,lambda  -- parameters for GJR GARCH
%     N -- # of time steps.
%
%       April 22, 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F0 = 0.5;
phi = alpha+lambda*F0+beta;
hbar = w/(1-phi);

[z,q] = zq(m_h, 0.6);
z = z(:);
q = q(:);
h1 = w+beta*h0 +h0*(alpha+lambda.*(z<0)).*z.^2;

n = 1:N;
% 1st moment
Ymom1 = n.*mu;
Ymom1 = Ymom1(:);
Ymom2 = zeros(N,1);
% second moment
for n = 1:N
    tmp = n*hbar +(1-phi)^(-1).*(1-phi^n).*(h1-hbar);
    Ymom2(n) = q'*tmp;    
end


%%  generate nodes by the fourth order moments
mu_X = Ymom1;
var = Ymom2;
% approximate the third moment
k3 = -lambda*ones(N,1);
% approximate the fourth moment
tmp = 3*hmom2(1)/(hmom1(1)^2);
k4 = tmp*ones(N,1);


G = [mu_X, var, k3, k4]';
nodes_Xt = Treenodes_JC_X(G,N,m_x,gamma_x);

end