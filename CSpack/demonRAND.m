% demon compressed sensing problems with random data 
clc; clear; close all; addpath(genpath(pwd));

n       = 10000;  
m       = ceil(0.25*n); 
s       = ceil(0.05*n); 

T       = randperm(n,s);  
xopt    = zeros(n,1);
xopt(T) = (0.1+rand(s,1)).*sign(randn(s,1));  
A       = randn(m,n)/sqrt(m);   
b       = A(:,T)*xopt(T)+0.00*randn(m,1);  

t       = 2; 
solver  = {'NHTP', 'GPNP', 'IIHT', 'PSNP', 'NL0R', 'MIRL1'};
out     = CSsolver(A,[],b,n,s,solver{t}); 

fprintf(' Objective of xopt:       %.2e\n', norm(A*xopt-b)^2/2);
fprintf(' Objective of out.sol:    %.2e\n', out.obj);
fprintf(' Sparsity of out.sol:     %2d\n', nnz(out.sol));
fprintf(' Computational time:      %.3fsec\n',out.time); 
if s<=1e3; RecoverShow(xopt,out.sol,[1000 500 500 250],1); end
