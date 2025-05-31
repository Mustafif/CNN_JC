function [imp_vol,V0, it] = impvol(S0, K, T,r,V,index,N,m,gamma, tol, itmax)
%
% compute the implied volatility of the American option by willow tree
% through the bisection method
%
% Input
%   S0 -- intial value of stock price
%    K -- strike prices, matrix
%    T -- maturities, vector
%    r -- interest rate
%    V -- American option prices
%    N -- # of time steps of willow tree
%    m -- # of tree nodes ate each time step
%   gamma -- gamma for sampling z
%
% Output
%   imp_vol -- implied volatilities of the American options
%
%
n = length(T);
k = size(K,1);
z = zq(m,gamma);
imp_vol = zeros(k,n);
for i = 1:n  % each maturity
    T_i = T(i);  % Extract current maturity as scalar
    [P,q]=gen_PoWiner(T_i,N,z);
    for j = 1:k
        V0 = 10000;
        it = 0;
        a = 0;
        b = 1;
        sigma = (a+b)/2;
        
        % Store target price for clarity
        target_price = V(j,i);
        
        while abs(V0-target_price) > tol && it < itmax
            Xnodes = nodes_Winer(T_i,N,z,r,sigma);
            nodes = S0.*exp(Xnodes);
            
            % Price American option with current tree
            V0 = American(nodes,P,q,r,T_i,S0,K(j,i),index);
            
            % Update bisection bounds
            if V0 > target_price
                b = sigma;
            else
                a = sigma;
            end
            sigma = (a+b)/2;
            it = it + 1;
        end
        imp_vol(j,i) = sigma;
    end
end
