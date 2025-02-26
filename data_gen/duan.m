function [hd, q] = duan(h0, beta, alpha, omega, mh, gamma_h)
    % Generate standard normal nodes
    [z, q] = zq(mh, gamma_h);
    z = z';

    % Compute conditional variances using Duan’s formula
    hd = omega + beta * h0 + alpha * z.^2;
    hd = sort(hd);

    % Compute integration points
    intPoints = (hd(1:end-1) + hd(2:end)) / 2;
    numHt = length(hd);

    % Compute bounds safely (avoid complex values)
    sqrtTerm = max((intPoints - omega - beta * h0) / alpha, 0);  % Clamp to zero
    upBound = real(sqrt(sqrtTerm));  % Remove small imaginary parts
    lowBound = -upBound;

    % Check for invalid values
    if any(~isfinite(upBound)) || any(~isfinite(lowBound))
        error('Error: upBound or lowBound contains NaN or Inf values.');
    end

    % Compute transition probabilities
    if alpha > 0
        prob = normcdf(upBound) - normcdf(lowBound);
        prob = [0; prob; 1];
        q = diff(prob)';
    elseif alpha < 0
        prob = 1 - (normcdf(upBound) - normcdf(lowBound));
        prob = [ones(1, numHt); prob; zeros(1, numHt)];
        q = (prob(1:end-1, :) - prob(2:end, :))';
    end
    
    % Ensure q is real and finite
    q = real(q(:));
    if any(~isfinite(q))
        error('Error: q contains NaN or Inf values.');
    end
end
