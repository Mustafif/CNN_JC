clear all
% tic


T = [30, 60, 90, 180, 360]./252;
m = linspace(0.8, 1.2, 9);
% T = [90];
% m = [0.85];
N = 504; %100;
delta=T/N;
% h0=(0.2^2)/252;
r = 0.05/252;
S0 = 100;
K = 100;
%
% alpha=1.33e-6;
% beta=0.586;
% gamma=484.69;
% omega=4.96e-6;
% lambda = 1/2;
alpha = 1.33e-6;
beta = 0.8;
omega = 1e-6;
gamma = 5;
lambda = 0.2;
% alpha = 1.33e-6;
% beta = 0.85;
% omega = 1e-6;
% gamma = 100.0;
% lambda = 0.5;
dt = T/N;


% generate willow tree
m_h = 6;
m_ht = 6;
m_x = 30;
gamma_h = 0.6;
gamma_x = 0.8;


% generate MC paths for GARCH model
M = 1;
numPoint = N+1;
Z = randn(numPoint+1,M);
% Z1 = randn(numPoint,numPath);
% ht = nan(numPoint+1,numPath);
% ht(1,:) = h0*ones(1,numPath);
% Xt(1,:) = log(S0)*ones(1,numPath);
% for i=2:numPoint
%     ht(i,:) = omega+alpha*(Z(i-1,:)-gamma*sqrt(ht(i-1,:))).^2+beta*ht(i-1,:);
%     Xt(i,:) = Xt(i-1,:)+(r-0.5*ht(i,:))+sqrt(ht(i,:)).*Z(i,:);
% end
% ht(i+1,:) = omega+alpha*(Z(i,:)-gamma*sqrt(ht(i,:))).^2+beta*ht(i,:);
% S = exp(Xt);
[S, h0] = mcHN(M, N,S0, Z, r, omega, alpha, beta, gamma, lambda);
%h0 = (0.2^2)/2;
day_per_contract = 5;
day = 1;

S0 = S(end + (-day_per_contract + day), :);
t = T - (day - 1);
for i = 1:length(T)
    if t(i) <= 0
        % Calculate how many full cycles have expired
        %expired_cycles = floor(abs(t(i)) / T(i)) + 1;

        % Restore maturity by applying the correct number of cycles
        t(i) = T(i) - mod(abs(t(i)), T(i));
    end
end

T_len = length(T);
m_len = length(m);
dataset = zeros(12, T_len * m_len * 2);

%
% Construct the willow tree for ht
%
[hd,qhd] = genhDelta(h0, beta, alpha, gamma, omega, m_h, gamma_h);
[nodes_ht] = TreeNodes_ht_HN(m_ht, hd, qhd, gamma_h,alpha,beta,gamma,omega,N+1);
%[P_ht_N, P_ht] = Prob_ht(nodes_ht,h0,alpha,beta,gamma,omega);


%
% construct the willow tree for Xt
[nodes_Xt,mu,var,k3, k4] = TreeNodes_logSt_HN(m_x,gamma_x,r,hd,qhd,S0,alpha,beta,gamma,omega,N);
[q_Xt,P_Xt,tmpHt] = Prob_Xt(nodes_ht,qhd, nodes_Xt, S0,r, alpha,beta,gamma,omega);
nodes_S = exp(nodes_Xt);
itmax = 60;
tol = 1e-4;
idx = 1;

for i = 1:T_len
    for j = 1:m_len
        K = m(j)*S0  + 0.01*S0*i*j; % Strike price
        [V_C, ~] = American(nodes_S,P_Xt,q_Xt,r,t(i),S0,K,1);
        [V_P, ~] = American(nodes_S,P_Xt,q_Xt,r,t(i),S0,K,-1);
         % [impl_c, ~] = impvol(S0, K, t(i),r,V_C,1,N,m_x,gamma_x, tol, itmax);
         % [impl_p,  ~] = impvol(S0, K, t(i),r,V_P,-1,N,m_x,gamma_x, tol, itmax);
         % impl_c = impl_c * sqrt(252);
         % impl_p = impl_p * sqrt(252);
        [impl_c, ~, ~] = impVol_HN(r, lambda,omega, beta, alpha, gamma,h0,S0, K, t(i), N, m_h, m_x, 1);
        [impl_p, ~, ~] = impVol_HN(r, lambda,omega, beta, alpha, gamma,h0,S0, K, t(i), N, m_h, m_x, -1);
        % Store call option data
        dataset(:, idx) = [S0; K/S0; r; t(i); 1; alpha; beta; omega; gamma; lambda; impl_c;V_C];
        idx = idx + 1;

        % Store put option data
        dataset(:, idx) = [S0; K/S0; r; t(i); -1; alpha; beta; omega; gamma; lambda;impl_p; V_P];
        idx = idx + 1;
    end
end
% Save the dataset to a CSV file
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'sigma', 'V'}';
filename = 'impl_demo.csv';
dataset_ = [headers'; num2cell(dataset')];
writecell(dataset_, filename);
disp(['Dataset saved as ', filename]);

% Filter the dataset to include only call options (corp == 1)
% dataset = dataset(:, dataset(:, 5) == 1);

% Extract strike prices and implied volatilities for call options
% strike_prices = dataset(:, 2)' .* dataset(:, 1)';  % Adjust based on column where strike prices are stored
% impl_vols = dataset(:, 11)';  % Adjust based on the column where implied volatilities are stored
%
% % Plot the volatility curve for call options
% figure;
% plot(strike_prices, impl_vols, 'b-', 'LineWidth', 2);
% title('Implied Volatility Curve for Call Options');
% xlabel('Strike Price / S0');
% ylabel('Implied Volatility');
% grid on;
% legend('Call Option Implied Volatility');


% % Show the plot
% hold off;



%
% Price the European options.
%
% E_WT = []; A_WT=[];
% E_MC = []; A_MC =[];
% Benchmark = [];
% for kk =1:1
%     Benchmark = [];
%     % compute option values
%
%     for k= 1:31
%         % K = 0.8*S0;
%         K=0.9*S0+0.01*S0*k;
%         % price by the willow tree
%         tic
%         time0 = toc;
%
%         tic;
%         %priceE = European(nodes_S,P_Xt,q_Xt,r,T,K,1);
%         timeE = time0+toc;
%
%         tic;
%         CorP = -1;
%         priceA(k) = American(nodes_S,P_Xt,q_Xt,r,T,S0,K,CorP);
%         timeA = time0+toc;
%         [sig(k), PriA(k), PricA0(k)] = impVol_HN(r, lambda,omega, beta, alpha, gamma,h0,S0, K, T, N, m_h, m_x, CorP);
%         %[price_WT(k)]=American_GBM_Pricing(S0,K,r,sig(k),T,CorP,N,m_x,gamma_x);
%         % disp([priceE,priceA]);
%         % E_WT = [E_WT,[priceE;timeE]];
%         %E_WTT(kk,k) = priceE;
%         %    A_WT = [A_WT,[priceA;timeA]];
%
%         % Price by the MC method
%         V=max(S(end,:)-K,0);
%         priceE_MC=exp(-r*T)*mean(V);
%         allPriceE = V;
%         s_E = std(allPriceE);
%         E_MCC(kk,k) = priceE_MC;
%
%
%
%
%         %Price =HestonNandi(S0,K,h0,T,r);
%         %Benchmark = [Benchmark, Price];
%     end
%
% end
