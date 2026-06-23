%% ------------------ SCALABLE HESTON-NANDI GARCH PARALLEL ------------------
clear all; clc;

fprintf('=================================================================\n');
fprintf('   SCALABLE HESTON-NANDI GARCH - PARALLEL VERSION\n');
fprintf('=================================================================\n\n');

%% ------------------ SCALING PARAMETERS ------------------
CONTRACTS_PER_DAY = 1;     
NUM_DAYS = 1;              
BATCH_SIZE = 1;            % Adjust batch size for parallelism
NUM_PARAM_SETS = 1;         

total_contracts = CONTRACTS_PER_DAY * NUM_DAYS * 2 * NUM_PARAM_SETS;  

num_batches = ceil(total_contracts / BATCH_SIZE);

fprintf('SCALING CONFIGURATION:\n');
fprintf('  Contracts per day: %d\n', CONTRACTS_PER_DAY);
fprintf('  Number of days: %d\n', NUM_DAYS);
fprintf('  Total contracts: %d\n', total_contracts);
fprintf('  Parameter sets: %d\n', NUM_PARAM_SETS);
fprintf('  Batch size: %d\n', BATCH_SIZE);
fprintf('  Number of batches: %d\n', num_batches);

%% ------------------ MARKET PARAMETERS ------------------
tic;
T = [30,60,90,180,360];
N = 504;
r = 0.05/252;
S0 = 100;
K_base = 100;

moneyness_range = [0.8,1.2];
maturity_range = [0.1,1.0];

%% ------------------ GARCH PARAMETER GENERATION ------------------
alpha_range = [0.01, 0.15];      % ARCH parameter
beta_range = [0.75, 0.85];             % GARCH parameter
omega_range = [1e-7, 1e-6];            % Long-run variance level
gamma_range = [0, 0];             % Asymmetry parameter
lambda_range = [0.01, 0.15];             % Risk premium parameter
param_sets = zeros(NUM_PARAM_SETS,5);
valid_sets = 0;

while valid_sets < NUM_PARAM_SETS
    alpha = alpha_range(1) + diff(alpha_range)*rand();
    beta  = beta_range(1) + diff(beta_range)*rand();
    omega = omega_range(1) + diff(omega_range)*rand();
    gamma = gamma_range(1) + diff(gamma_range)*rand();
    lambda= lambda_range(1) + diff(lambda_range)*rand();
    
    if beta + alpha + lambda/2 < 0.999
        valid_sets = valid_sets + 1;
        param_sets(valid_sets,:) = [alpha,beta,omega,gamma,lambda];
    end
end

%% ------------------ WILLOW TREE & SOLVER PARAMETERS ------------------
m_h = 6; m_ht = 6; m_x = 30;
gamma_h = 0.6; gamma_x = 0.8;
itmax = 100; tol = 1e-6;

%% ------------------ PRE-GENERATE GARCH PATHS ------------------
M_paths = min(NUM_PARAM_SETS,50);
numPoint = N + 1;
Z_seeds = randn(numPoint+1, M_paths);

S_paths_all = cell(NUM_PARAM_SETS,1);
h0_paths_all = zeros(NUM_PARAM_SETS,1);

for p = 1:NUM_PARAM_SETS
    alpha_p = param_sets(p,1); beta_p = param_sets(p,2);
    omega_p = param_sets(p,3); gamma_p = param_sets(p,4);
    lambda_p = param_sets(p,5);
    
    seed_idx = mod(p-1,M_paths)+1;
    [S_path,h0_path] = mcGJR(1,N,S0,Z_seeds(:,seed_idx),r,omega_p,alpha_p,beta_p,lambda_p);
    S_paths_all{p} = S_path;
    h0_paths_all(p) = h0_path;
end

%% ------------------ ENSURE WORKERS SEE PATHS ------------------
currentFolder = pwd;
addpath(currentFolder);
addpath(genpath(currentFolder));
pctRunOnAll addpath(currentFolder);
pctRunOnAll addpath(genpath(currentFolder));

%% ------------------ PARALLEL BATCH PROCESSING ------------------
dataset_cells = cell(num_batches,1);
batch_counts = zeros(num_batches,1);

parfor batch = 1:num_batches
    contracts_in_batch = min(BATCH_SIZE, total_contracts - (batch-1)*BATCH_SIZE);
    local_data = zeros(12, contracts_in_batch);
    ptr = 1;
    
    for contract_idx = (batch-1)*BATCH_SIZE + 1 : (batch-1)*BATCH_SIZE + contracts_in_batch
        day = ceil(contract_idx/CONTRACTS_PER_DAY);
        param_idx = mod(contract_idx-1,NUM_PARAM_SETS)+1;
        alpha = param_sets(param_idx,1); beta = param_sets(param_idx,2);
        omega = param_sets(param_idx,3); gamma = param_sets(param_idx,4);
        lambda = param_sets(param_idx,5);
        S_path = S_paths_all{param_idx};
        h0_path = h0_paths_all(param_idx);
        S_current = S_path(end);
        
        rng(contract_idx);
        moneyness = moneyness_range(1) + diff(moneyness_range)*rand();
        maturity_years = maturity_range(1) + diff(maturity_range)*rand();
        K_strike = moneyness*S_current;
        T_maturity = maturity_years;
        
        % Build tree
        c = gamma + lambda + 0.5;
        % [hd,qhd] = genhDelta(h0_path,beta,alpha,c,omega,m_h,gamma_h);
        [nodes_ht, qht, hmom1, hmom2, hmom3, hmom4_app]= TreeNodes_ht_GJR(m_h, h0_path, gamma_h, omega, beta, alpha, lambda, N+1);
        [nodes_Xt, ~, ~, ~, ~] = TreeNodes_logSt_GJR(m_x,m_h, gamma_x, r, omega, alpha, lambda,beta,h0_path, N, hmom1, hmom2);
        nodes_Xt = nodes_Xt + log(S_current);

        [q_Xt,P_Xt,~] = Probility_Xt2(nodes_ht, qht, nodes_Xt, S_current, r, omega, beta, alpha, lambda);
        nodes_S = exp(nodes_Xt);
        
        % Price options
        [V_C,~] = American(nodes_S,P_Xt,q_Xt,r,T_maturity,S_current,K_strike,1);
        [V_P,~] = American(nodes_S,P_Xt,q_Xt,r,T_maturity,S_current,K_strike,-1);
        [impl_c,~,~] = impvol(S_current,K_strike,T_maturity,r,V_C,1,N,m_x,gamma_x,tol,itmax);
        [impl_p,~,~] = impvol(S_current,K_strike,T_maturity,r,V_P,1,N,m_x,gamma_x,tol,itmax);
        
        % Store call
        local_data(:,ptr) = [S_current;K_strike/S_current;r;T_maturity;1;alpha;beta;omega;gamma;lambda;impl_c;V_C];
        ptr = ptr + 1;
        % Store put
        local_data(:,ptr) = [S_current;K_strike/S_current;r;T_maturity;-1;alpha;beta;omega;gamma;lambda;impl_p;V_P];
        ptr = ptr + 1;
    end
    
    dataset_cells{batch} = local_data(:,1:ptr-1);
    batch_counts(batch) = ptr-1;
    fprintf('Batch %d complete with %d contracts\n', batch, ptr-1);
end

%% ------------------ SERIAL ASSEMBLY ------------------
total_cols = sum(batch_counts);
dataset = zeros(12,total_cols);

pos = 1;
for b = 1:num_batches
    cols = batch_counts(b);
    if cols>0
        dataset(:,pos:pos+cols-1) = dataset_cells{b};
        pos = pos + cols;
    end
end
elapsed = toc;
fprintf("Elapsed time: %6f\n", elapsed);
fprintf('=================================================================\n');
fprintf('PARALLEL DATASET GENERATION COMPLETE\n');
fprintf('Saved: varying_garch_parallel_%d.csv\n', total_contracts*2);
fprintf('Total contracts: %d\n', size(dataset,2));
fprintf('=================================================================\n');

% Save dataset
% headers = {'S0','m','r','T','corp','alpha','beta','omega','gamma','lambda','sigma','V'}';
% dataset_enhanced = [headers'; num2cell(dataset')];
% filename = sprintf('GJR_%dK.csv', (total_contracts)/1000);
% writecell(dataset_enhanced,filename);
