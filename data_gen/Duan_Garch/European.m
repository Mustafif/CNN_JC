function price = European(nodes,P,q,r,T,K,index)
% function price = European(nodes,q,r,T,K,index)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Introdutcion 
%         Given the willow tree including tree-nodes matrix and transition
%         probabilityvector, this function calculate the price of an
%         European options(both call and put).
%
%     Input
%         nodes (M*N matrix) : willow tree-nodes matrix
%         q (M*1 vector)     : q(i) represents the transition probability from initial rate
%         r (a scalar)       : risk free rate
%         T(a scalar)        : expiration
%         K (a scalar)       : exercise price
%         index(a scalar)    : flag for choice of options(1:call,-1:put)
%
%      Output
%         price              :the computed price of the option
%          
%      Implemented by
%         S.H. Huang  2018.5
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,N] = size(nodes);
V(:,N) = max(index*(nodes(:,N)-K),0);% the payoff at maturity date
% Backward induction
for i = N-1:-1:1;
    V(:,i) = exp(-r*T/N)*P(:,:,i)*V(:,i+1);
end
price = exp(-r*T/N)*q'*V(:,1); % European style
% price = exp(-r*T)*q'*V(:,N);