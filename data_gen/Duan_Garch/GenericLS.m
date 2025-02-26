function [price,s_A,x,y,z]= GenericLS(S0,K,r,T,SPaths,NSteps,NRepl,fhandles)
%% 功能最小二乘的蒙特卡洛模拟计算美式期权

dt = T/NSteps;
discountVet = exp(-r*dt*(1:NSteps)');
NBasis = length(fhandles);        %基函数个数
% alpha = zeros(NBasis,1);        %回归参数矩阵
% RegrMat = zeros(NRepl,NBasis);  %初始化设计矩阵。

% generate sample paths
%SPaths=AssetPaths_MCGBM(S0,r,sigma,T,NSteps,NRepl);
% SPaths(:,1) = [];               % 删除第一列

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