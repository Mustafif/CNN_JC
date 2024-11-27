function S0 = GenDaysData(day_num, r, T, m, alpha, beta, omega, gamma, lambda)
% market params: [r, T, m]
% garch params: [alpha, beta, omega, gamma, lambda]

% N = monte carlo time points, we are considering 2 years 
% which in trading days is 504 days 
N = 504;
% We will be simulating this under 1 path for the monte carlo simulation 
M = 1;

Z = randn(N + 1, M); % z_i ~ N(0, 1)

S_init = 100;

[S, h0] = mcHN(M, N, S_init, Z, r, omega, alpha, beta, gamma, lambda);

S0 = S(end+(-5+day_num), :);
T = T - (day_num - 1);
T_len = length(T);
m_len = length(m);
% V_cal = zeros(T_len, m_len);
% V_put = zeros(T_len, m_len);

call = 1;
% put = -1;
dataset = zeros(11, (m_len * 2) + (T_len*2));

for i = 1:T_len % each maturity 
    for j = 1:m_len % each moneyness
        K = m(j) * S0; % each strike
        [V_C, ~] = datagen2(T(i), r, S0, h0, K, alpha, beta, omega, gamma, lambda, call);
        [V_P, ~] = datagen2(T(i), r, K, h0, S0, alpha, beta, omega, gamma, lambda, call);

%         V_cal(i, j) = V_C;
%         V_put(i, j) = V_P;
        dataset(i, j) = [S0 K r T(i) 1 alpha beta omega gamma lambda V_C ]';
        dataset(i+1, j+1) = [S0 K r T(i) -1 alpha beta omega gamma lambda V_P ]';
    end
end

% if day_num == 1
%     S = round(S, 4);
%     T_S = array2table(S);
%     %T_S.T = repmat(T(:), 1, 1);
%     writetable(T_S, "Historical.csv");
% end


% Convert matrices into a table with labels
V_cal = round(V_cal, 4);
V_put = round(V_put, 4);

% % T_call: Call options table
% T_call = array2table(V_cal, 'VariableNames', strcat('K_', string(m)));
% 
% % Expand the Maturity vector to match the number of rows in the table
% T_call.T = repmat(T(:), 1, 1);  % Convert T to a column vector, match rows
% 
% % Assign the OptionType as 'Call' for all rows
% T_call.Type = repmat({'Call'}, size(V_cal, 1), 1);
% 
% % T_put: Put options table
% T_put = array2table(V_put, 'VariableNames', strcat('K_', string(m)));
% 
% % Expand the Maturity vector for the put options as well
% T_put.T = repmat(T(:), 1, 1);  % Convert T to a column vector, match rows
% 
% % Assign the OptionType as 'Put' for all rows
% T_put.Type = repmat({'Put'}, size(V_put, 1), 1);
% 
% % Combine the two tables (calls and put
% T = [T_call; T_put];
% 
% % Reorder columns so 'OptionType' comes first
% T = T(:, [{'Type', 'T'}, strcat('K_', string(m))]);
% 
% % Save the table to CSV
% filename = sprintf("Day%d.csv", day_num);
% writetable(T, filename);
end