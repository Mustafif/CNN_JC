%% Contracts
% contract is defined as a 1x19 matrix 
% 1 - r 
% 2-10 T
% 11-19 m

% for our stage 3 dataset, we will do 4 contracts 
c1 = [1.33e-6 0.8 1e-6 100 0.5];  
c2 = [2.5e-6 0.85 5e-7 50 0.3];  
c3 = [5e-6 0.7 2e-6 150 0.6];  
c4 = [1e-6 0.6 3e-6 200 0.4];  


ds1 = dataset_contract2(c1);
ds2 = dataset_contract2(c2);
ds3 = dataset_contract2(c3);
ds4 = dataset_contract2(c4);

ds = [ds1 ds2 ds3 ds4];

headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'V'}';
filename = 'stage3.csv';
dataset = [headers'; num2cell(ds')];
writecell(dataset, filename);
disp(['Dataset saved as ', filename]);


