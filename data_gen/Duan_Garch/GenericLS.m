function [price,s_A,x,y,z]= GenericLS(S0,K,r,T,SPaths,NSteps,NRepl,fhandles)
%% ������С���˵����ؿ���ģ�������ʽ��Ȩ

dt = T/NSteps;
discountVet = exp(-r*dt*(1:NSteps)');
NBasis = length(fhandles);        %����������
% alpha = zeros(NBasis,1);        %�ع��������
% RegrMat = zeros(NRepl,NBasis);  %��ʼ����ƾ���

% generate sample paths
%SPaths=AssetPaths_MCGBM(S0,r,sigma,T,NSteps,NRepl);
% SPaths(:,1) = [];               % ɾ����һ��

CashFlows = max(0, K - SPaths(:,NSteps));
ExerciseTime = NSteps*ones(NRepl,1);
for step = NSteps-1:-1:1
    InMoney = find(SPaths(:,step) < K);
    XData = SPaths(InMoney,step);
    RegrMat = zeros(length(XData), NBasis);
    for k=1:NBasis
        RegrMat(:, k) = feval(fhandles{k}, XData);
    end
    YData = CashFlows(InMoney).*discountVet(ExerciseTime(InMoney)-step);
    alpha = RegrMat \ YData;
    IntrinsicValue = K - XData;
    ContinuationValue = RegrMat * alpha;
    Index = find(IntrinsicValue > ContinuationValue);
    ExercisePaths = InMoney(Index);
    CashFlows(ExercisePaths) = IntrinsicValue(Index);
    ExerciseTime(ExercisePaths) = step;
end 
price = max(K-S0, mean(CashFlows.*discountVet(ExerciseTime)));
s_A = std(max(K-S0,CashFlows.*discountVet(ExerciseTime)));
 x=CashFlows;
 y=ExerciseTime;
 z=discountVet(ExerciseTime);