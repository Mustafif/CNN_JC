function S = mcHN(M, N,S0, h0,Z, r, omega, alpha, beta, gamma, lambda)

numPoint = N+1;
ht = nan(numPoint, M);
Xt = nan(numPoint, M);
ht(1,:) = h0 * ones(1, M);
Xt(1,:) = log(S0) * ones(1, M);
for i = 2:numPoint
    ht(i,:) = omega + alpha * (Z(i-1,:) - gamma * sqrt(ht(i-1,:))).^2 + beta * ht(i-1,:);
    Xt(i,:) = Xt(i-1,:) + (r - 0.5 * ht(i,:)) + sqrt(ht(i,:)) .* Z(i,:);
end
S = exp(Xt);
end