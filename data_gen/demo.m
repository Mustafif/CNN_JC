clear all
% tic

T = 100;
N = 100;
delta=T/N;
h0=(0.2^2)/252;
r = 0.05/250;
S0 = 100;
%K = 100;
%
alpha=1.33e-6;
beta=0.586;
gamma=484.69;
omega=4.96e-6;
lambda = 1/2;
dt = T/N;

% generate willow tree
m_h = 6;
m_ht = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;

% generate MC paths for GARCH model
numPath = 100000;
numPoint = N+1;
Z = randn(numPoint+1,numPath);
Z1 = randn(numPoint,numPath);
ht = nan(numPoint+1,numPath);
ht(1,:) = h0*ones(1,numPath);
Xt(1,:) = log(S0)*ones(1,numPath);
for i=2:numPoint
    ht(i,:) = omega+alpha*(Z(i-1,:)-gamma*sqrt(ht(i-1,:))).^2+beta*ht(i-1,:);
    Xt(i,:) = Xt(i-1,:)+(r-0.5*ht(i,:))+sqrt(ht(i,:)).*Z(i,:);
end
ht(i+1,:) = omega+alpha*(Z(i,:)-gamma*sqrt(ht(i,:))).^2+beta*ht(i,:);
S = exp(Xt);

% 
% Construct the willow tree for ht
%
[hd,qhd] = genhDelta(h0, beta, alpha, gamma, omega, m_h, gamma_h);
[nodes_ht] = TreeNodes_ht_HN(m_ht, hd, qhd, gamma_h,alpha,beta,gamma,omega,N+1);
[P_ht_N, P_ht] = Prob_ht(nodes_ht,h0,alpha,beta,gamma,omega);

%
% construct the willow tree for Xt
[nodes_Xt,mu,var,k3, k4] = TreeNodes_logSt_HN(m_x,gamma_x,r,hd,qhd,S0,alpha,beta,gamma,omega,N);
[q_Xt,P_Xt,tmpHt] = Prob_Xt(nodes_ht,qhd, nodes_Xt, S0,r, alpha,beta,gamma,omega);
nodes_S = exp(nodes_Xt);

sigA = zeros(31, 1);
PriA = zeros(31, 1);
PriA0 = zeros(31, 1);

parfor k = 1:31
    CorP = -1;
    K = 0.9*S0+0.1*S0*k;

    [sigA(k), PriA(k), PriA0(k)] = impVol_HN(r, lambda, omega, beta, alpha, gamma, h0, S0, K, T, N, m_h, m_x, CorP);
end


