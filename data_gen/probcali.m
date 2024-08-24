function pc=probcali(p,y,mu,sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Introduction
%          Given a row vector and a column vector, two scalars, i.e., mean and variance,
%           this function calibrate transition probability to keep mean and variance right
%
%  Input
%       p (1*N vector)  : original transition probability, row vector
%       y (N*1 vector)  : varibles, column vector
%      mu (a scalar)     : mean
%     sigma (a scalar) : standard volatility
%
%  Output
%        pc (1*N vector) : probability vector after calibration
%
%  References
%   1. W.Xu, L.Lu,two-factor willow tree method for convertibel bond pricing
%      with stochastic interest rate and default risk, 2016.
%
%  Implemented by
%          L.Lu 2016.12.15
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m=length(p);
a=p*y-mu;
b=p*y.^2-mu^2-sigma^2;
[~,pind]=sort(p);% from to min to max
x=zeros(1,m);
wm=pind(m); %max
w2=pind(m-1);
w1=pind(m-2);
y1=y(w1); %min
y2=y(w2); %middle
ym=y(wm); %max probability
x(wm)=(-b+a*(y1+y2))/(ym^2-y2^2-(y1+y2)*(ym-y2));
x(w1)=(-a-x(wm)*(ym-y2))/(y1-y2);
x(w2)=-(x(wm)+x(w1));

pc=p+x;
pc=max(pc,0);
pc= pc/sum(pc);