%
% Test pour fonction hhh.c
%

clear all;

mu=0; sd=1; ka3=-1.2; ka4=7.0;

[a,b,d,c,itype,ifault]=f_hhh(mu,sd,ka3,ka4);
[a b d c itype ifault]

[mom]=f_john_mom([a; b; c; d;],itype,ka3,ka4)

 z=randn(1000000,1);
 if itype==1
    u=(z-a)./b;
    gi=exp(u);
    x=c+d.*gi;
 elseif itype==2
    u=(z-a)./b;
    gi=((exp(u)-exp(-u))./2);
    x=c+d.*gi;
 elseif itype==3
    u=(z-a)./b;
    gi=1./(1+exp(-u));
    x=c+d.*gi;
 elseif itype==4
    u=(z-a)./b;
    gi=u;
    x=gi;
 end
 
 [mu sd ka3 ka4]
 [mean(x) std(x) skewness(x) kurtosis(x)]



