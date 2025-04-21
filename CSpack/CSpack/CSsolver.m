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
%     out.sp:    Sparsity level of out.sol
%     out.time:  CPU time
%     out.iter:  Number of iterations
%     out.obj:   Objective function value at out.sol 
% =========================================================================
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!! ! 
% =========================================================================

warning off; 
if  nargin<6  
    disp(' Inputs is not enough !!! \n');
    return;
elseif nargin<7
    pars = []; 
end

if  isa(A,'function_handle') && isempty(At)
    disp(' A is a function handle. Its transpose is missing!\n'); 
    disp(' Please input its transpose At!\n');
    return;
end

if  isempty(s)
    if  ismember(solver,{'NHTP', 'GPNP', 'IIHT'})
        disp(' Since no sparsity level is provided, change solver to <PSNP> \n');  
        solver = 'PSNP';  
    end
else
    s      = ceil(s); 
end


Solver = str2func(solver);   
if ~isfield(pars,'disp');  pars.disp = 1; end 

data.A  = A;
data.At = At;
data.b  = b;
if ~isfield(pars,'lambda')
    if  isa(A,'function_handle')
        lambda  = 0.0025*norm(At(b),'inf');
    else
        lambda  = 0.0025*norm(b'*A,'inf');
    end
    lambda_NL0R = 50*lambda;
else
    lambda      = pars.lambda;
    lambda_NL0R = lambda;
end
 
switch solver
    case 'NHTP'; out = Solver(data,n,s,pars);
    case 'GPNP'; out = Solver(data,n,s,pars);
    case 'IIHT'; out = Solver(data,n,s,pars);
    case 'PSNP'; out = Solver(data,n,lambda,pars);
    case 'NL0R'; out = Solver(data,n,lambda_NL0R,pars);
    case 'MIRL1';out = Solver(data,n,lambda,pars);
end
thresh = 0; 
if  out.error > 1e-4 && ~isa(A,'function_handle')
    if  isempty(s)
        sx     = sort(abs(out.sol(out.sol~=0))); 
        it     = find(normalize(sx(2:end)./sx(1:end-1))>6);     
        if ~isempty(it) && it>1
            thresh = sx(it(1)); 
        end 
    else
         sx     = maxk(abs(out.sol),s+1); 
         thresh = sx(end);
    end 
end 

if ~isa(A,'function_handle')
    T    = find(abs(out.sol)>thresh); 
    x    = zeros(n,1);            
    x(T) = linsolve(A(:,T),b);
    out.sol = x;
    out.obj = norm(A*x-b)^2/2;
end

fprintf(' -------------------------------------\n'); 
if out.obj<1e-10
  fprintf(' A global minimizer may be found\n');
  fprintf(' since (1/2)||Ax-b||^2 = %5.2e\n',out.obj);   
  fprintf(' -------------------------------------\n');
end

end

