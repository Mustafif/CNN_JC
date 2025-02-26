function p = Prob(current_ht,next_ht, beta_0,beta_1, beta_2, theta, lambda)
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
            upBound(:,i) = theta+lambda +sqrt((intPoints-beta_0-beta_1*nowHt)./(beta_2.*nowHt));
            lowBound(:,i) = theta+lambda -sqrt((intPoints-beta_0-beta_1*nowHt)./(beta_2.*nowHt));
        end        
            prob = normcdf(real(upBound))-normcdf(real(lowBound));
            prob = [zeros(1,numHt);prob;ones(1,numHt)];
            p = (diff(prob))';        
    end