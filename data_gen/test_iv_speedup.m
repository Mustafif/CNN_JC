% Test IV calculation speedup
S0 = 100; K = 100; T = 0.25; r = 0.05; V_target = 5; 
N = 50; m_x = 50; gamma_x = 0.5;

fprintf('Testing IV calculation speedup...\n');

% Test WITHOUT cached matrices
tic;
[iv1, ~, ~] = impvol_fzero(S0, K, T, r, V_target, 1, N, m_x, gamma_x);
time1 = toc;
fprintf('Without cache: IV=%f, time=%.3f sec\n', iv1, time1);

% Test WITH cached matrices
z = zq(m_x, gamma_x);
[P, q] = gen_PoWiner(T, N, z);
tic;
[iv2, ~, ~] = impvol_fzero(S0, K, T, r, V_target, 1, N, m_x, gamma_x, z, P, q);
time2 = toc;
fprintf('With cache: IV=%f, time=%.3f sec\n', iv2, time2);
fprintf('Speedup: %.1fx\n', time1/time2);
