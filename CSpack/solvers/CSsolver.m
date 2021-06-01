function out = CSsolver(data,n,solver,pars)
% This solver solves compressive sensing (CS) in one of the following forms:
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
% where s << n is the given sparsity and lambda>0, mu>0 
%       A\in\R{m by n} the measurement matrix
%       b\in\R{m by 1} the observation vector 
% =========================================================================
% Inputs:
%   data  : A structure (REQUIRED)
%           (data.A, data.b) if A is a matrix 
%           (data.A, data.b, data.At) if A is a function handle
%           i.e., Ax = data.A(x); A'y = data.At(y); 
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
%           pars.eta   --  A positive scalar for 'NHTP' (default, 1)                       
%           pars.disp  --  Results of each step are displayed or not (default,1)
%           pars.maxit --  Maximum number of iterations (default,2000) 
%           pars.tol   --  Tolerance of the halting condition (default,1e-6)
%           ------------------Particular for NL0R -------------------------
%           pars.tau   --  A positive scalar for 'NL0R' (default, 1)  
%           pars.lam   --  An initial penalty parameter (default, 0.1)
%           pars.obj   --  A predefined lower bound of f(x),(default,1e-20)
%           pars.rate  --  A positive scalar to adjust lam, (default,  0.5) 
%           ------------------Particular for IIHT -------------------------
%           pars.neg   --  Compute SCCS (default, 0)
%                          Compute SCCS with a non-negative constraint,x>=0
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


switch  nargin 
    case 1 
         disp(' No enough inputs. No problems will be solverd!'); 
         return;
    case 2
         solver = 'NL0R';
end

if ~isfield(data,'A')
    fprintf('<data.A> is missing, unable to run the solver ...');
    return
else
    if  isa(data.A,'function_handle') && ~isfield(data,'At') 
        fprintf('<data.At> is missing, unable to run the solver ...');
        return
    end
end

if  ~isfield(data,'b')
    fprintf('<data.b> is missing, unable to run the solver ...');
    return
end
 
if ismember(solver, {'NHTP','IIHT'})
   if ~isfield(pars,'s')
       fprintf(' The sparsity level is missing for solver %s! \n',solver);
       pars.s = input(' Input the sparsity level [e.g.,0.01*n], pars.s = ');
       if isempty(pars.s) 
       fprintf(' The sparsity level is isempty, replace %s by NL0R\n',solver); 
       solver = 'NL0R'; 
       else
       pars.s = ceil(pars.s);
       end
   end
end

Solver = str2func(solver);   
if ~isfield(pars,'disp');  pars.disp = 1; end 
switch solver
    case 'NHTP' ; out = Solver(data,n,ceil(pars.s),pars);
    case 'IIHT' ; out = Solver(data,n,ceil(pars.s),pars);
    case 'NL0R' ; out = Solver(data,n,pars);
    case 'MIRL1'; out = Solver(data,n,pars);
end

end

