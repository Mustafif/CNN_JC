function [nodes_ht,p, hmom1,hmom2,hmom3, hmom4] = TreeNodes_ht_GJR(m_h, h0, gamma_h,w,beta,alpha, lambda, N)
%
% compute the first four moments of h_t of the GJR GARCH(1,1) model
% and construct all tree nodes of h_t under Q measure
%
%       X_t = mu+ sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = w + beta*h_t+(alpha+lambda*I_{z_t<0})h_t^z_t^2
%
%
%  Input
%     m_h  -- number of tree nodes of h_t
%     h0 -- initial value
%     gamma_h -- parameter for zq
%     w, beta, alpha, lambda-- parameters for GJR GARCH
%     N -- # of time steps.
%
%  Output
%    modes_ht -- tree nodes for h_t
%    p -- probability of h_t at t=1 given h0
%    hmom1, hmom2 -- first two moments of ht 
%
%       May 29, 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate h1
[z,q] = zq(m_h,gamma_h);
q = q(:);
z = z(:);
h1= w+beta*h0+(alpha+lambda.*(z<0)).*h0.*z.^2;

% compute the conditional moments of h_t, t>2
k_z = 3;
F0 = 0.5;
phi = alpha+lambda*F0+beta;
hbar = w/(1-phi);

hmom1 = zeros(N+1,1);
hmom2 = zeros(N+1,1);
hmom3 = zeros(N+1,1);
hmom4 = zeros(N+1,1);


gamma=phi.^2+(k_z-1)*(alpha+lambda*F0).^2+k_z*lambda^2*F0*(1-F0);
c1 = (w^2+2*w*phi*hbar)./(1-gamma);
c2 = 2*w.*phi.*(h1-hbar)./(phi-gamma);
c3 = c1+c2;
c6 = 15.*(alpha^3+3*alpha*lambda*(alpha+lambda)*F0+lambda^3*F0) +3*beta*gamma ...
        -beta^2*(2*beta+3*(alpha+lambda*F0));
c7 = 105*(alpha^4+F0*(lambda^4+4*(alpha^3*lambda+alpha*lambda^3)+6*alpha^2*lambda^2)) ...
    +beta^4+4*(15*beta*(alpha^3+F0*(lambda^3+3*(alpha^2*lambda+alpha*lambda^2))) ...
    +beta^3*(alpha+lambda*F0)) +6*k_z*beta^2*(alpha^2+lambda^2*F0+2*alpha*lambda*F0);

hmom1(1) = q'*h1;
hmom2(1) = q'*h1.^2;
hmom3(1) = q'*h1.^3;
hmom4(1)= q'*h1.^4;
    
% first four moments
for n = 2:N+1
    % 1st
    tmp  = hbar+phi.^(n-1).*(h1-hbar);
    hmom1(n) = q'*tmp;
    % 2nd
    tmp = c1+c2*phi.^(n-1)+(h1.^2-c3).*gamma.^(n-1);
    hmom2(n) = q'*tmp;
    
    % 3rd
    tmp = c6.^(n-1).*h1.^3;
    tmp1 = c7^(n-1).*h1.^4;
    for i = 0:n-2
        tmp = tmp +c6^i.*(w^3+3*w^2.*phi.*hmom1(n-i-1)+3.*w.*gamma.*hmom2(n-i-1));
        tmp1 = tmp1 + c7.^i.*(w.^4+4.*w.^3.*phi.*hmom1(n-i-1)+6.*w.^2.*gamma.*hmom2(n-i-1) ...
            + 4.*w.*c6.*hmom3(n-i-1));
    end
        hmom3(n) = q'*tmp;
        hmom4(n) = q'*tmp1;
      
end


%% generate nodes by 4th order moments
mu = hmom1(1:end);
var = hmom2(1:end)-mu.^2;
temp = sqrt(var);
k3 = 1./(temp.^3).*(hmom3(1:end)-3*mu.*hmom2(1:end)+2*mu.^3);
k4 = 1./(temp.^4).*(hmom4(1:end)-4*mu.*hmom3(1:end)+6*(mu.^2).*hmom2(1:end)-3*mu.^4);

k3 = k3(2)*ones(N+1,1);
k4 = k4(2)*ones(N+1,1);

G= [mu, var,k3,k4]';
[nodes_ht, p] = Treenodes_JC_h(G,N+1,m_h,gamma_h);

%p = Prob(h0, nodes_ht(:,1), beta_0,beta_1, beta_2, theta, lambda);
p = p(:);
end




