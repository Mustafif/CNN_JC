%% Contracts
% contract is defined as a 1x19 matrix 
% 1 - r 
% 2-10 T
% 11-19 m

% for our stage 2 dataset, we will do 5 contracts 
c1 = [(0.05)/504 ...
    [5, 10, 21, 42, 63, 126, 180, 252, 360]...
    linspace(0.7, 1.3, 9)];
% c2 = [0.03 ...
%     [5, 12, 26, 48, 63, 123, 160, 252, 352]...
%     linspace(0.65, 1.25, 9)];
% c3 = [0.03 ...
%     [7, 18, 26, 40, 69, 120, 161, 255, 300]...
%     linspace(0.7, 1.3, 9)];
% c4 = [0.03 ...
%     [5, 10, 21, 42, 63, 126, 180, 252, 360]...
%     linspace(0.65, 1.3, 9)];

ds1 = dataset_contract(c1);
% ds2 = dataset_contract(c2);
% ds3 = dataset_contract(c3);
% ds4 = dataset_contract(c4);

% ds = [ds1 ds2 ds3 ds4];
ds = ds1;
headers = {'S0', 'm', 'r', 'T', 'corp', 'alpha', 'beta', 'omega', 'gamma', 'lambda', 'V'}';
filename = 'stage1b.csv';
dataset = [headers'; num2cell(ds')];
writecell(dataset, filename);
disp(['Dataset saved as ', filename]);


