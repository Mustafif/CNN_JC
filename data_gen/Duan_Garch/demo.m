clear all
% tic

T = 152;
N = 152;
delta=T/N;
r = 0.05/365;
S0 = 100;
K = 100;
%
beta_0 = 10^(-5);
beta_1 = 0.70;
beta_2 = 0.1;
lambda = 0.25;
theta = 0.25;
dt = T/N;


h = beta_0/(1-beta_1-beta_2*(1+(theta+lambda)^2));
h0 = h*1.2;
% generate willow tree
m_h = 20;
m_x = 50;
gamma_h = 0.8;
gamma_x = 0.8;

% generate MC paths for GARCH model
numPath = 10000;
numPoint = N+1;
Z = randn(numPoint+1,numPath);
Z1 = randn(numPoint,numPath);
ht = nan(numPoint+1,numPath);
ht(1,:) = h0*ones(1,numPath);
Xt(1,:) = log(S0)*ones(1,numPath);
for i=2:numPoint 
    ht(i,:) = beta_0+ beta_1*ht(i-1,:)+beta_2*ht(i-1,:).*(Z(i-1,:)-theta-lambda).^2;
    Xt(i,:) = Xt(i-1,:)+(r-0.5*ht(i,:))+sqrt(ht(i,:)).*Z(i,:);
    
end
ht(i+1,:) = beta_0+ beta_1*ht(i,:)+beta_2*ht(i,:).*(Z(i,:)-theta-lambda).^2;
S = exp(Xt);

% comute the first four moments of ht

mu_h = sum(ht')/numPath;
k2_h = sum((ht').^2)/numPath;
k3_h = sum((ht').^3)/numPath;
k4_h = sum((ht').^4)/numPath;

% mu_h = mu_h(2:end);
% k2_h = k2_h(2:end);
% k3_h = k3_h(2:end);
% k4_h = k4_h(2:end);

var_h = k2_h-mu_h.^2;
std_h = sqrt(var_h);
skew_h = 1./(std_h.^3).*(k3_h-3*mu_h.*k2_h+2*mu_h.^3);
kurt_h = 1./(std_h.^4).*(k4_h-4*mu_h.*k3_h+6*(mu_h.^2).*k2_h-3*mu_h.^4);

mu_X = sum((Xt-log(S0))')/numPath;
k2_X = sum((Xt-log(S0))'.^2)/numPath;
k3_X = sum((Xt-log(S0))'.^3)/numPath;
k4_X = sum((Xt-log(S0))'.^4)/numPath;

% mu_X = mu_X(2:end);
% k2_X = k2_X(2:end);
% k3_X = k3_X(2:end);
% k4_X = k4_X(2:end);

var_X = k2_X-mu_X.^2;
std_X = sqrt(var_X);
skew_X = 1./(std_X.^3).*(k3_X-3*mu_X.*k2_X+2*mu_X.^3);
kurt_X = 1./(std_X.^4).*(k4_X-4*mu_X.*k3_X+6*(mu_X.^2).*k2_X-3*mu_X.^4);
% 
% Construct the willow tree for ht
%
[nodes_ht,qht, hmom1, hmom2, hmom3, hmom4_app] = TreeNodes_ht_D(m_h, h0, gamma_h,beta_0, beta_1,beta_2, theta, lambda,N+1);
%[P_ht_N, P_ht] = Probility_ht(nodes_ht,h0,alpha,beta,gamma,omega);

%
% construct the willow tree for Xt
% G = [mu_X(2:end); var_X(2:end); skew_X(2:end); kurt_X(2:end)];
% nodes_Xt = Treenodes_JC_X(G,N,m_x,gamma_x);
[nodes_Xt,mu,var,k3, k4] = TreeNodes_logSt_D(m_x,gamma_x,r, h0, beta_0,beta_1, beta_2,theta, lambda,N, hmom1, hmom2);
nodes_Xt = nodes_Xt+log(S0);
% [nodes_Xt,mu,var,k3, k4] = TreeNodes_logSt_HN_New(m_x,gamma_x,r,hd,qhd,S0,alpha,beta,gamma,omega,N);
[q_Xt,P_Xt,tmpHt] = Probility_Xt2(nodes_ht,qht, nodes_Xt, S0,r, beta_0,beta_1,beta_2,theta, lambda);
nodes_S = exp(nodes_Xt);


% Price the European options.

E_WT = []; A_WT=[];
E_MC = []; A_MC =[];
Benchmark = [];
for kk =1:1
    Benchmark = [];    
    % compute option values
    
    for k= 1:31
        K=0.8*S0+0.01*S0*k;
        % price by the willow tree
        tic
        time0 = toc;
        
        tic;
        priceE = European(nodes_S,P_Xt,q_Xt,r,T,K,1);
        timeE = time0+toc;
        
        %     tic;
            priceA = American(nodes_S,P_Xt,q_Xt,r,T,S0,K,-1);
        %     timeA = time0+toc;
        % disp([priceE,priceA]);
        % E_WT = [E_WT,[priceE;timeE]];
        E_WTT(kk,k) = priceE;
        A_WTT(kk,k) = priceA;
        
        % Price by the MC method
        V=max(S(end,:)-K,0);
        priceE_MC=exp(-r*T)*mean(V);
        allPriceE = V;
        s_E = std(allPriceE);
        E_MCC(kk,k) = priceE_MC;
        
        SPaths=S(2:end,:);
        SPaths=SPaths';
        fhandles={@(x)x,@(x)2*x.^2-1,@(x)4*x.^3-3*x};
        [priceA,s_A]= GenericLS(S0,K,r,T,SPaths,N,numPath,fhandles);
        A_MCC(kk,k) = priceA;

%         Price =HestonNandi(S0,K,h0,T,r);
%         Benchmark = [Benchmark, Price];
       
    end
   
   
end

