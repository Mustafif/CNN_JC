function p = Prob(current_ht,next_ht, alpha, beta,gamma, omega)
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
            upBound(:,i) = gamma*sqrt(nowHt)+sqrt((intPoints-beta*nowHt-omega)/alpha);
            lowBound(:,i) = gamma*sqrt(nowHt)-sqrt((intPoints-beta*nowHt-omega)/alpha);
        end
        if alpha>0
            prob = normcdf(real(upBound))-normcdf(real(lowBound));
            prob = [zeros(1,numHt);prob;ones(1,numHt)];
            p = (diff(prob))';
        elseif alpha<0  % alpha���������Ʋ�����С��0�����һ��ȷ��
            prob = 1-(normcdf(real(upBound))-normcdf(real(lowBound)));
            prob = [ones(1,numHt);prob;zeroes(1,numHt)];
            p = (prob(1:end-1,:)-prob(2:end,:))';
        end
        
    end