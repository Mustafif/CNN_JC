function [nodes,q]=Treenodes_JC_h(G,N,M,gamma)
[z,q,~,~] = zq(M,gamma);   % generate discrete sampling values
nodes=zeros(M,N);  % Initialization
itype=zeros(N,1);

% Determine the type of Johnson Curve parameters,a,b,c,d and the function,g
% by f_hhh process

for i=1:N
    mu=G(1,i);
    sd=sqrt(G(2,i));
    ka3=G(3,i);
    ka4=G(4,i);
    [a,b,d,c,itype(i),~]=f_hhh(mu,sd,ka3,ka4);
    
    % Transform the discrete values of standard normal
    % distributioed variable into our required underlying asset
    % by Johnson formula under different cases.
    
    if itype(i)==1 %type 1 lognormal
        u=(z(:)-a)./b;
        gi=exp(u(:));
        x=c+d.*gi(:);
    elseif itype(i)==2 %type 2 unbounded
        u=(z(:)-a)./b;
        gi=((exp(u(:))-exp(-u(:)))./2);
        x=c+d.*gi(:);
    elseif itype(i)==3 %type 3 bounded
        u=(z(:)-a)./b;
        gi=1./(1+exp(-u(:)));
        x=c+d.*gi(:);
    elseif itype(i)==4 %type 4 normal
        u=(z(:)-a)./b;
        gi=u(:);
        x=c+d.*gi(:);
    else
        x = linspace(mu-4*sd, mu+4*sd, M);
        x = x(:);
    end
    
    % Check for negative values and handle both first iteration and subsequent iterations
    if i == 1
        if sum(x<0) > 0
            x = linspace(mu*0.5, mu*1.5, M);  % Use mu-based range for first iteration
        else
            x = linspace(mu-4*sd, mu+4*sd, M);
        end
    else
        if sum(x<0) > 0
            x = linspace(nodes(1,i-1)*0.9, nodes(end,i-1)*1.1, M);
        else
            x = linspace(mu-4*sd, mu+4*sd, M);
        end
    end
    
    nodes(:,i)=sort(x); %generate the i-th column of the underlying asset matrix
end
end