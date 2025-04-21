To implement this solver, please 

    Step 1. run startup.m first to add the path 
    Step 2. run demonXXXX.m to solve different problems

This package contains 5 solvers for compressive sensing problems 
based on the algorithms described in the following papers:

NHTP------------------------------------------------------------------------
    S. Zhou, N. Xiu, and H. Qi, 
    Global and quadratic Convergence of Newton hard-thresholding pursuit, 
    Journal of Machine Learning Research, 22(12):1-45, 2021
GPNP------------------------------------------------------------------------
    S. Zhou,
    Gradient projection Newton pursuit for sparsity constrained optimization, 
    Applied and Computational Harmonic Analysis, 61:75-100, 2022
PSNP------------------------------------------------------------------------
    S Zhou, X Xiu, Y Wang, and D Peng, 
    Revisiting Lq ( 0 ≤ q < 1 ) norm regularized optimization, 
    arXiv:2306.14394, 2023 
NL0R------------------------------------------------------------------------
    S. Zhou, L. Pan, and N. Xiu, 
    Newton method for l_0 regularized optimization,
    Numerical Algorithms, 2021
IIHT------------------------------------------------------------------------
    L. Pan, S. Zhou, N. Xiu, and H. Qi, 
    A convergent iterative hard thresholding for nonnegative sparsity optimization, 
    Pacific Journal of  Optimization, 13(2), 325-353, 2017
MIRL1-----------------------------------------------------------------------
    S. Zhou, N. Xiu, Y. Wang, L. Kong and H. Qi, 
    A Null-space-based weighted  l1 minimization approach to compressed sensing, 
    Information and Inference: A Journal of the IMA, vol. 5(1): 76-102, 2016.

Please credit them if you use the code for your research.

===========================================================================
function out = CSsolver(A,At,b,n,s,solver,pars)
% This solver solves compressive sensing (CS) in one of the following forms
%
% 1) The sparsity constrained compressive sensing (SCCS)
%
%         min_{x\in R^n} 0.5||Ax-b||^2  s.t. ||x||_0<=s
%
% 2) The L0 regularized compressive sensing (LqCS)
%
%         min_{x\in R^n} 0.5||Ax-b||^2 + lambda * ||x||_q^q,  0<=q<1 
%
% 3) The reweighted L1-regularized compressive sensing (RL1CS)
%
%         min_{x\in R^n} 0.5||Ax-b||^2 + lambda||Wx||_1
%
% where s << n is the given sparsity and lambda>0 
%       A\in\R{m by n} the measurement matrix
%       b\in\R{m by 1} the observation vector 
%       W\in\R{n by n} is a diagonal matrix to be updated iteratively
% =========================================================================
% Inputs:
%   A  :     The measurement matrix, A\in\R{m by n}              (REQUIRED)
%   At :     The transpose of A and can be [] if A is a matrix   (REQUIRED)
%            But At is REQUIRED if A is a function handle 
%            i.e., A*x = A(x); A'*y = At(y); 
%   b:       The observation vector  b\in\R{m by 1}              (REQUIRED)
%   n:       Dimension of the solution x,                        (REQUIRED)
%   s:       The sparsity level, if unknown, put it as []        (REQUIRED)
%   solver:  A text string, can be one of                        (REQUIRED)
%            {'NHTP','GPNP','PSNP','NL0R','IIHT','MILR1'}
%
%           --------------------------------------------------------------------------------
%                    |  'NHTP'   |  'GPNP'   |  'PSNP'   |  'NL0R'   |  'IIHT'   |  'MIRL1'   
%           --------------------------------------------------------------------------------
%           Model    |   SCCS    |   SCCS    |   LqRCS   |   L0RCS   |   SCCS    |   RL1CS     
%           Method   | 2nd-order | 2nd-order | 2nd-order | 2nd-order | 1st-order | 1st-order  
%           Sparsity | required  | required  |  no need  |  no need  | required  |  no need
%           --------------------------------------------------------------------------------  
%
%   pars  : ----------------For all solvers -------------------------------
%           pars.x0     --  Starting point of x       (default, zeros(n,1))                     
%           pars.disp   --  =1, show results for each step      (default,1)
%                           =0, not show results for each step
%           pars.maxit  --  Maximum number of iterations     (default, 2e3) 
%           pars.tol    --  Tolerance of stopping criteria   (default,1e-6)
%           ----------------Particular for NHTP ---------------------------
%           pars.eta    --  A positive scalar for 'NHTP'       (default, 1)  
%                           Tuning pars.eta may improve solution quality.
%           ----------------Particular for PSNP ---------------------------
%           pars.q      --  Decide Lq norm                  (default,  0.5)  
%           pars.lambda --  An initial penalty parameter    (default,  0.1)
%           pars.obj    --  A predefined lower bound of f(x)(default,1e-20)
%           ----------------Particular for NL0R ---------------------------
%           pars.tau    --  A positive scalar for 'NL0R'    (default,    1)  
%           pars.lambda --  An initial penalty parameter    (default,  0.1)
%           pars.obj    --  A predefined lower bound of f(x)(default,1e-20)
%           ----------------Particular for IIHT ---------------------------
%           pars.neg    --  =0, Compute SCCS without x>=0       (default,0)
%                           =1, Compute SCCS with x>=0
% =========================================================================
% Outputs:
%     out.sol:   The sparse solution x
%     out.sp:    Sparsity level of Out.sol
%     out.time:  CPU time
%     out.iter:  Number of iterations
%     out.obj:   Objective function value at Out.sol 
% =========================================================================
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!! ! 
% =========================================================================

% Below is one example that you can run
% =========================================================================
clc; clear; close all; addpath(genpath(pwd));

n       = 10000;  
m       = ceil(0.25*n); 
s       = ceil(0.05*n); 

T       = randperm(n,s);  
xopt    = zeros(n,1);
xopt(T) = (0.1+rand(s,1)).*sign(randn(s,1));  
A       = randn(m,n)/sqrt(m);   
b       = A(:,T)*xopt(T)+0.00*randn(m,1);  

t       = 1; 
solver  = {'NHTP', 'GPNP', 'IIHT', 'PSNP', 'NL0R', 'MIRL1'};
out     = CSsolver(A,[],b,n,s,solver{t}); 

fprintf(' Objective of xopt:       %.2e\n', norm(A*xopt-b)^2/2);
fprintf(' Objective of out.sol:    %.2e\n', out.obj);
fprintf(' Sparsity of out.sol:     %2d\n', nnz(out.sol));
fprintf(' Computational time:      %.3fsec\n',out.time); 
if s<=1e3; RecoverShow(xopt,out.sol,[1000 500 500 250],1); end
