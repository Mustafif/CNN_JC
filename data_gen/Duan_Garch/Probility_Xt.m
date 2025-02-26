function [q_Xt,P_Xt_N, tmpHt] = Probility_Xt(nodes_ht,q_ht,nodes_Xt, S0,r, beta_0,beta_1,beta_2, theta,lambda)
%
% compute the transition probabilities X_t of the Heston-Nadi GARCH model and
% construct all tree nodes of X_t and h_t
%
%       X_t = r-0.5*h_t + sqrt(h_t)z_t    z_t~~N(0,1)
%       h_{t+1} = beta_0+beta_1*h_t+beta_2*h_t*(z_t-theta-lambda)^2
%
%
%  Input
%     nodes_ht  -- tree nodes of ht
%     q_ht -- probabiliies of h_1 given h0
%     nodes_Xt -- tree nodes of Xt
%     S0 -- intial value of St
%     r -- interest rate
%     beta_0,beta_1,beta_2, theta,lambda -- parameters for Duan's GARCH
% 
%  Output
%     q_Xt -- transition probabilities from X1 to X0.
%     P_Xt_N -- transition probability matrix of Xt, 3-d array
%    
%
%       April 22, 2022
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X0 = log(S0);
m_h = size(nodes_ht,1);
[X_len, N] = size(nodes_Xt);

% Compute q_Xt for t=1
Xt = nodes_Xt(:, 1);
m_x = X_len;
cur_ht = nodes_ht(:,1);

mu = r-0.5*cur_ht;
std = sqrt(cur_ht);
dx = Xt-X0;
intX = [-inf; (dx(1:end-1)+dx(2:end))/2; inf];
tmpP = zeros(m_h, m_x);
Ph = zeros(m_h, m_x, N);
Ph_XXh_h = zeros(m_h,m_x,m_x);
next_ht = nodes_ht(:,2);
for i = 1:m_h
    % compute P(X_i^1|X^0)
    p = normcdf(intX, mu(i), std(i));
    p = diff(p);
    tmpP(i,:) = p(:)';
    % compute P(h_j^2|X_i^1,X^0)
    ht = cur_ht(i);
    z =[-inf; (intX(2:end-1)-r+0.5*ht)./sqrt(ht); inf];
    intH = [beta_0 + beta_1*ht+1e-16; (next_ht(1:end-1)+next_ht(2:end))/2; 1];
    zhup1 = real(-sqrt(intH(1:end-1)-beta_0-beta_1*ht)/sqrt(beta_2*ht)+theta+lambda);
    zhlow1 = real(-sqrt(intH(2:end)-beta_0-beta_1*ht)/sqrt(beta_2*ht)+theta+lambda);
    zhup2 = real(sqrt(intH(2:end)-beta_0-beta_1*ht)/sqrt(beta_2*ht)+theta+lambda);
    zhlow2 = real(sqrt(intH(1:end-1)-beta_0-beta_1*ht)/sqrt(beta_2*ht)+theta+lambda);
    for j = 1:m_x
        tmplow1 = max(zhlow1, z(j));
        tmplow2 = max(zhlow2, z(j));
        tmpup1 = min(zhup1, z(j+1));
        tmpup2 = min(zhup2, z(j+1));
        % compute P(h_k^2|X_j^1, X^0, h_i^1)
        ph1 = max(normcdf(tmpup1)-normcdf(tmplow1),0) + ...
            max(normcdf(tmpup2) - normcdf(tmplow2), 0);
%         ph1 = max(normalcdf(tmpup1)-normalcdf(tmplow1),0) + ...
%             max(normalcdf(tmpup2) - normalcdf(tmplow2), 0);
        ph1 = ph1(:)';
        Ph_XXh_h(:,j,i) = ph1./tmpP(i,j);
    end    
end

q_Xt = q_ht'*tmpP;
q_Xt = q_Xt(:);

for i = 1:m_h
    Ph_YY = tmpP(i,:).*q_ht(i)./(q_Xt');
    Ph(:,:,1) = Ph(:,:,1)+Ph_XXh_h(:,:,i).*repmat(Ph_YY,m_h,1);
end
for i = 1:m_x
    Ph(:,i,1) = Ph(:,i,1)/sum(Ph(:,i,1));
end

P_Xt = zeros(m_x,N);
P_Xt(:,1) = q_Xt;
tmpHt(:,1) = Ph(:,:,1)*P_Xt(:,1);
%
% compute transition probability matrices [p_ij]^n
%
P_Xt_N = zeros(m_x,m_x, N-1);
for n = 1:N-1
    next_ht = nodes_ht(:,n+2);
    cur_ht = nodes_ht(:,n+1);
    Xt = nodes_Xt(:,n+1);
    mu = r-0.5*cur_ht;
    std = sqrt(cur_ht);
    Ph_XXX  = zeros(m_h,m_x,m_x,m_h);
    for i = 1:m_x  %X_i^n
        
        cur_Xt = nodes_Xt(i, n);
        dx = Xt-cur_Xt;
        intX = [-1000; (dx(1:end-1)+dx(2:end))/2; 1000];
        for j = 1:m_h % h_j^n+1
            % compute P(X^n+1|X_i^n, h_j^n+1)
            p = normcdf(intX, mu(j), std(j));
            p = diff(p);
            tmpP(j,:) = p(:)';
            % compute P(h^n+2)|X^n+1, X_i^n)
            ht = cur_ht(j);
            z =[-1000; (intX(2:end-1)-r+0.5*ht)./sqrt(ht); 1000];
            intH = [beta_0 + beta_1*ht+1e-16; (next_ht(1:end-1)+next_ht(2:end))/2; 1];
            zhup1 = real(-sqrt(intH(1:end-1)-beta_0-beta_1*ht)/sqrt(beta_2*ht)+theta+lambda);
            zhlow1 = real(-sqrt(intH(2:end)-beta_0-beta_1*ht)/sqrt(beta_2*ht)+theta+lambda);
            zhup2 = real(sqrt(intH(2:end)-beta_0-beta_1*ht)/sqrt(beta_2*ht)+theta+lambda);
            zhlow2 = real(sqrt(intH(1:end-1)-beta_0-beta_1*ht)/sqrt(beta_2*ht)+theta+lambda);
            for k = 1:m_x  % X_j^n+1
                tmplow1 = max(zhlow1, z(k));
                tmplow2 = max(zhlow2, z(k));
                tmpup1 = min(zhup1, z(k+1));
                tmpup2 = min(zhup2, z(k+1));
                % compute P(h^n+2|X_k^n+1, X_i^n, h_j^n+1)
                id1 = (tmplow1<tmpup1);
                id2 = (tmplow2<tmpup2);
                
                if tmpP(j,k)<1e-4  % tmpP(j,k) == 0
                    num = sum(id1)+sum(id2);
                    Ph_XXX(id1, k,i,j) = 1/num;
                    Ph_XXX(id2,k,i,j) = 1/num;
                    % Ph_XXX(:,k,i,j) = Ph_XXX(:,k,i,j)/sum(Ph_XXX(:,k,i,j));
                else
                    ph1 = zeros(m_h,1);
                    ph1(id1) = normalcdf(tmpup1(id1))-normalcdf(tmplow1(id1));
                    ph1(id2) = ph1(id2)+normalcdf(tmpup2(id2))-normalcdf(tmplow2(id2));
                    tmp = ph1./tmpP(j,k);
                    Ph_XXX(:,k,i,j) = tmp/sum(tmp);
                end
 
            end  % end for k
        end  %end for j
        tmp = Ph(:,i,n)'*tmpP; 
        P_Xt_N(i,:,n) = tmp/sum(tmp);
        
    end  % end for i
    
    P_Xt(:,n+1) = (P_Xt(:,n)'*P_Xt_N(:,:,n))';
    
   
    Ph_XXh_h = zeros(m_h,m_x,m_x);
    for e = 1:m_x
        tmp = reshape(Ph(:,e,n),1,1,m_h);
        tmp = repmat(tmp, m_h,m_x);
        Ph_XXh_h(:,:,e) = sum(reshape(Ph_XXX(:,:,e,:), m_h,m_x,m_h).*tmp,3);
    end
    for d = 1: m_x
        tmp = P_Xt_N(:,d,n).*P_Xt(:,n)/P_Xt(d,n+1);
        sumtmp = zeros(m_h,1);
        for e = 1:m_x
            sumtmp = sumtmp + Ph_XXh_h(:,d,e).*tmp(e);
        end
        Ph(:,d,n+1) = sumtmp;
    end

     tmpHt(:,n+1) = Ph(:,:,n+1)*P_Xt(:,n+1);
end

end