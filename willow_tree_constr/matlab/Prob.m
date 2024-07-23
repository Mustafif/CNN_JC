function p = Prob(current_ht,next_ht, alpha, beta,gamma, omega)
        %================================================
        % 计算概率转移矩阵
        % current_ht是一个列向量（或一个点），next_ht也是一个列向量
        % 返回p是一个行向量（或一个矩阵）
        %================================================
        intPoints = (next_ht(1:end-1)+next_ht(2:end))/2;
        %         intPoints = [-inf;intPoints;inf];
        numHt = length(current_ht);
        upBound = nan(length(intPoints),numHt);
        lowBound = nan(length(intPoints),numHt);
        for i=1:numHt
            nowHt = current_ht(i);
            upBound(:,i) = gamma*sqrt(nowHt)+sqrt((intPoints-beta*nowHt-omega)/alpha);
            lowBound(:,i) = gamma*sqrt(nowHt)-sqrt((intPoints-beta*nowHt-omega)/alpha);
        end
        if alpha>0
            prob = normcdf(real(upBound))-normcdf(real(lowBound));
            prob = [zeros(1,numHt);prob;ones(1,numHt)];
            p = (diff(prob))';
        elseif alpha<0  % alpha可能有限制不可以小于0，需进一步确认
            prob = 1-(normcdf(real(upBound))-normcdf(real(lowBound)));
            prob = [ones(1,numHt);prob;zeroes(1,numHt)];
            p = (prob(1:end-1,:)-prob(2:end,:))';
        end
        
    end