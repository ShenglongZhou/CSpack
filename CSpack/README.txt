To implement this solver, please 

    [1] run startup.m first to add the path;
    [2] run demonXXXX.m to solve different problems

This package contains 4 solvers for compressive sensing problems 
based on the algorithms described in the following papers:

NHTP------------------------------------------------------------------------------------
    S. Zhou, N. Xiu and H. Qi, 
    Global and Quadratic Convergence of Newton Hard-Thresholding Pursuit, 
    Journal of Machine Learning Research, 22(12):1-45, 2021
NL0R-------------------------------------------------------------------------------------
    S. Zhou, L. Pan and N. Xiu, 
    Newton Method for l_0 Regularized Optimization,
    Numerical Algorithms, 2021
IIHT--------------------------------------------------------------------------------------
    L. Pan, S. Zhou, N. Xiu and H. Qi, 
    A convergent iterative hard thresholding for nonnegative sparsity optimization, 
    Pacific Journal of  Optimization, 13(2), 325-353, 2017
MIRL1------------------------------------------------------------------------------------
    S. Zhou, N. Xiu, Y. Wang, L. Kong and H. Qi, 
    A Null-space-based weighted  l1 minimization approach to compressed sensing, 
    Information and Inference: A Journal of the IMA, vol. 5(1): 76-102, 2016.

Please give credits to them if you use the code for your research.

===========================================================================
% The citation of the solver <CSsolver> takes the form of
%
%                out = CSsolver(data,n,solver,pars)
%
% It solves compressive sensing (CS) in one of the following forms:
%
% 1) The sparsity constrained compressive sensing (SCCS): 
%
%         min_{x\in R^n} 0.5||Ax-b||^2  s.t. ||x||_0<=s
%
% 2) The L0 regularized compressive sensing (L0CS)
%
%         min_{x\in R^n} 0.5||Ax-b||^2 + lambda * ||x||_0 
%
% 3) The reweighted L1-regularized compressive sensing (RLCS)
%
%         min_{x\in R^n} 0.5||Ax-b||^2 + mu||w.*x||_1
%
% where s << n is the given sparsity and lambda>0, mu>0.  
% =========================================================================
% Inputs:
%   data  : A triple structure (data.A, data.At, data.b) (REQUIRED)
%           data.A, the measurement matrix, or a function handle @(x)A(x)
%           data.At = data.A',or a function handle @(x)At(x)
%           data.b, the observation vector 
%   n     : Dimension of the solution x, (REQUIRED)
%   solver: a text string, can be one of {'NHTP','NL0R','IIHT','MILR1'}
%           ------------------------------------------------------------
%                    |   'NHTP'   |   'NL0R'   |   'IIHT'   |   'MIRL1'   
%           ------------------------------------------------------------
%           Model    |    SCCS    |    L0CS    |     SCCS   |    RLCS     
%           Method   |  2nd-order |  2nd-order |  1st-order |  1st-order  
%           Sparsity |  required  |  optional  |  required  |  optional
%           ------------------------------------------------------------             
%   pars  : pars.x0    --  Starting point of x (default, zeros(n,1))
%           pars.s     --  Sparsity of x, an integer between 1 and n-1  
%                          This is REQUIRED for 'NHTP' and 'IIHT'
%           pars.tau   --  A positive scalar (default, 1) 
%                          This is vaild for 'NHTP' and 'NL0R'
%           pars.x0    --  Starting point of x (default, zeros(n,1))
%           pars.disp  --  Results of each step are displayed or not (default,1)
%           pars.draw  --  A graph is drawn or not (default,0) 
%           pars.maxit --  Maximum number of iterations (default,2000) 
%           pars.tol   --  Tolerance of the halting condition (default,1e-6)
%           ------------------Particular for NL0R -------------------------
%           pars.lam   --  An initial penalty parameter (default, 0.1)
%           pars.obj   --  A predefined lower bound of f(x), (default,1e-20)
%           pars.rate  --  A positive scalar to adjust lam, (default, 0.5) 
%           ------------------Particular for IIHT -------------------------
%           pars.neg   --  Compute SCCS (default, 0)
%                          Compute SCCS with a non-negative constraint, x>=0
% =========================================================================
% Outputs:
%     Out.sol:   The sparse solution x
%     Out.sp:    Sparsity level of Out.sol
%     Out.time:  CPU time
%     Out.iter:  Number of iterations
%     Out.obj:   Objective function value at Out.sol 
% =========================================================================
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!! ! 
% =========================================================================

% Below is one example that you can run
% =========================================================================
clc; clear; close all;

n       = 20000;  
m       = ceil(n/4); 
s       = ceil(0.05*n); 

% generate the testing data (data.A, data.At, data.b)
I0      = randperm(n); 
I       = I0(1:s);
xopt    = zeros(n,1);
xopt(I) = randn(s,1); 
data.A  = randn(m,n)/sqrt(m);
data.At = data.A';                
data.b  = data.A(:,I)*xopt(I);  

% choose one of the following four solvers                              
solver  = {'NHTP', 'NL0R', 'IIHT', 'MIRL1'};
pars.s  = s; % required for solvers 'NHTP' and 'IIHT' 
out     = CSsolver(data,n,solver{1},pars);

% results output and recovery display 
fprintf(' CPU time:     %.3fsec\n',out.time);
fprintf(' Objective:    %.2e\n', out.obj);
fprintf(' Sample size:  %dx%d\n', m,n);
ReoveryShow(xopt,out.sol,[1000, 550,500 200],1)
