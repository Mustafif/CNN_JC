function GenDaysData(day_num)
% N = monte carlo time points, we are considering 2 years 
% which in trading days is 504 days 
N = 504;
% We will be simulating this under 1 path for the monte carlo simulation 
M = 1;

Z = randn(N + 1, M); % z_i ~ N(0, 1)

S_init = 100;
r = 0.03; % fixed 3% 

alpha = 1.33e-6;
beta = 0.8;
omega = 1e-6;
gamma = 100;
lambda = 0.5;
[S, h0] = mcDuan(M, N, S_init, Z, r, omega, alpha, beta,lambda);

S0 = S(end+(-5+day_num), :);
T = [5, 10, 21, 42, 63, 126];
T = T - (day_num - 1);
m = linspace(0.8, 1.2, 9);
T_len = length(T);
m_len = length(m);
V_cal = zeros(T_len, m_len);
V_put = zeros(T_len, m_len);

call = 1;
% put = -1;

parfor i = 1:T_len
    for j = 1:m_len
        K = m(j) * S0;
        [V_C, ~] = datagen3(T(i), r, S0, h0, K, alpha, beta, omega, gamma, lambda, call);
        [V_P, ~] = datagen3(T(i), r, K, h0, S0, alpha, beta, omega, gamma, lambda, call);

        V_cal(i, j) = V_C;
        V_put(i, j) = V_P;
    end
end


% Convert matrices into a table with labels
V_cal = round(V_cal, 4);
V_put = round(V_put, 4);

% T_call: Call options table
T_call = array2table(V_cal, 'VariableNames', strcat('K_', string(m)));

% Expand the Maturity vector to match the number of rows in the table
T_call.T = repmat(T(:), 1, 1);  % Convert T to a column vector, match rows

% Assign the OptionType as 'Call' for all rows
T_call.Type = repmat({'Call'}, size(V_cal, 1), 1);

% T_put: Put options table
T_put = array2table(V_put, 'VariableNames', strcat('K_', string(m)));

% Expand the Maturity vector for the put options as well
T_put.T = repmat(T(:), 1, 1);  % Convert T to a column vector, match rows

% Assign the OptionType as 'Put' for all rows
T_put.Type = repmat({'Put'}, size(V_put, 1), 1);

% Combine the two tables (calls and put
T = [T_call; T_put];

% Reorder columns so 'OptionType' comes first
T = T(:, [{'Type', 'T'}, strcat('K_', string(m))]);

% Save the table to CSV
filename = sprintf("Duan_Day%d.csv", day_num);
writetable(T, filename);
end 

for i = 1:1:5
    GenDaysData(i);
end