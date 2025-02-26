function [price,V]=American(nodes,P,q,r,T,S0,K,index)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Introdutcion 
%         Given the willow tree including tree-nodes matrix and transition
%         probability matrices, this function calculate the price of an
%         American put option.
%
%      Input
%          nodes (M*N matrix)         willow tree-nodes matrix
%          P (M*M*(N-1) matrix)     P£¨i,j,k) represents the transition probability from i-th node
%                                                        at time t_k to j-th node at t_{k+1}.
%          q (M*1 vector)                 q(i) represents the transition probability from initial rate
%           r (a scalar)                          risk free rate
%           T(a scalar)                          expiration
%           S0 (a scalar)                       initial stock price
%           K (a scalar)                         exercise price
%           index(a scalar)                  +1 for call and -1 for put
%
%       Output
%          price             the computed price of the option
%
%         Implemented by
%                   G.Wang  2016.12
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,N]=size(nodes);
V(:,N) = max(index*(nodes(:,N)-K),0);% the payoff at maturity date
% Backward induction
for i = N-1:-1:1;
    V(:,i) = max(max(index*(nodes(:,i)-K),0),exp(-r*T/N)*P(:,:,i)*V(:,i+1));
end
price = max(max(index*(S0-K),0),exp(-r*T/N)*q'*V(:,1));   
