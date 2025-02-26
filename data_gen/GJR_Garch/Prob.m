function p = Prob(current_ht,next_ht, beta_0,beta_1, beta_2, theta, lambda)
        %================================================
        % �������ת�ƾ���
        % current_ht��һ������������һ���㣩��next_htҲ��һ��������
        % ����p��һ������������һ������
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