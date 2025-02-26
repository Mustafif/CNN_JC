function Y = MLE_obj_NH(x)
%
%  Objective function of MLE of NH-GARCH model given the log return with
%  paramters in x
%
% INPUT
%   x -- paramters of NH-GARCH model
%     x(1) -- w
%     x(2) -- beta
%     x(3) -- alpha
%     x(4) -- gamma
%     x(5) -- lambda 
%   r -- data of log returns
%
% OUTPUT
%  Y -- function value of MLE 
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load("rofGS.mat");
%r = r(1:300);
n = length(r);
mu = mean(r);
h0 = var(r);

epsilon = zeros(n,1);
h = zeros(n+1,1);
h(1) = h0;
for i = 1:n
    epsilon(i) = (r(i)-mu+0.5)/sqrt(h(i));
    h(i+1) = x(1)+x(2)*h(i)+x(3)*(epsilon(i)-x(4)*sqrt(h(i)))^2;
end
Y = 0.5*sum(log(h(2:n+1))+(r -mu+0.5).^2./h(2:n+1));