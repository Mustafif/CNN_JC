function [x,Y, flag, output] = MLE_NHGARCH(dataname, tol, maxit, x0)
%
%  Calibrate the parameters of NH-GARCH model from the historical log
%  return data under P measure.
% 
%       X_t = mu-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = omega +alpha(z_t-gamma*sqrt(h_t))^2+beta h_t
%
%
%  Input
%    dataname -- data file name of the daily log return in .Mat
%    tol -- stopping criterion of MLE method
%    maxit -- maximum iteration numbers
%    x0 -- initial value of parameters
%     x0(1) -- w
%     x0(2) -- beta
%     x0(3) -- alpha
%     x0(4) -- gamma
%     x0(5) -- lambda 
%
%  Output
%    w, beta, alpha, gamma -- parameters of the NH-GARCH model.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r = load(dataname);

% options = optimoptions('fminunc');
% options.Algorithm = 'trust-region';

fun = "MLE_obj_HN";
[x,Y, flag, output] = fmincon("MLE_obj_NH", x0,[],[],[],[],zeros(4,1),[]);
w = x(1);
beta = x(2);
alpha = x(3);
gamma = x(4);
lambda = 0.5;
end
