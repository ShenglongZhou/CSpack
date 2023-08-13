% demon compressed sensing problems with random data 
clc; clear; close all;

n       = 10000;  
m       = ceil(n/4); 
s       = ceil(0.05*n); 

% generate the testing data (data.A, data.At, data.b)
I       = randperm(n,s);  
xopt    = zeros(n,1);
xopt(I) = randn(s,1); 
data.A  = normalization(randn(m,n),3);       
data.b  = data.A(:,I)*xopt(I);  

% choose one of the following four solvers  
t       = 1; 
solver  = {'NHTP', 'GPNP', 'IIHT', 'NL0R', 'MIRL1'};
pars.s  = s; % required for solvers 'NHTP' and 'IIHT' 
out     = CSsolver(data,n,solver{t},pars); 

% results output and recovery display 
fprintf(' CPU time:     %.3fsec\n',out.time);
fprintf(' Objective:    %.2e\n', out.obj);
fprintf(' Sample size:  %dx%d\n', m,n);
RecoverShow(xopt,out.sol,[1000 500 500 250], 1)


