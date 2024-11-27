S = zeros(5, 1);

r = 0.03;
T = [5, 10, 21, 42, 63, 126];
m = linspace(0.8, 1.2, 9);

alpha = 1.33e-6;
beta = 0.8;
omega = 1e-6;
gamma = 100;
lambda = 0.5;

for i = 1:1:5
   S(i) = GenDaysData(i, r, T, m, alpha, beta, omega, gamma, lambda);
end