x0(1) = 4.96e-6;
x0(2) = 0.586;
x0(3) = 1.33e-6;
x0(4) = 484.69;
%x0(5) = 1/2;

x = x0+randn(1)*0.5*x0;
%x = rand(4,1);
x = x0;
dataname = "rofGS.mat";
tol = 1e-6;
maxit = 100;
Y = MLE_obj_NH(x);
[x1, Y1, flag, output] = MLE_NHGARCH(dataname, tol, maxit, x0);