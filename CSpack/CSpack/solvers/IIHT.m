function out = IIHT(data,n,s,pars)
% A solver for sparsity constrained CS:
%           min_{x\in R^n} 0.5||Ax-b||^2,  s.t. ||x||_0<=s,
% or sparsity and non-negative constrained CS:
%           min_{x\in R^n} 0.5||Ax-b||^2,  s.t. ||x||_0<=s, x>=0,
% where s is the given sparsity, which is << n.  
%       A\in\R{m by n} the measurement matrix
%       b\in\R{m by 1} the observation vector 
% ========================================================================= 
% Inputs:
%     data    : A structure (required)
%               (data.A, data.b) if A is a matrix 
%               (data.A, data.b, data.At) if A is a function handle
%               i.e., Ax = data.A(x); A'y = data.At(y); 
%     n       : Dimension of the solution x, (required)
%     s       : Sparsity level of x, an integer between 1 and n-1, (required)           
%     pars:     Parameters are all OPTIONAL
%               pars.neg    --  = 0. Compute sparsity constrained model (default)
%                               = 1. Compute sparsity and non-negative constrained model 
%               pars.x0      --  Starting point of x, pars.x0=zeros(n,1) (default)
%               pars.disp    --  Display or not results for each iteration (default, 1)
%               pars.maxit   --  Maximum number of iterations (default,2000) 
%               pars.tol     --  Tolerance of the halting condition (default,1e-6)
%
% Outputs:
%     out.sol:           The sparse solution 
%     out.obj:           f(out.sol)
%     out.sp:            Sparsity level of out.sol
%     out.time           CPU time
%     out.iter:          Number of iterations
% ========================================================================= 
% This solver was created based on the algorithm proposed by  
% Pan, L., Zhou, S., Xiu, N. & Qi, H.D. (2017). A convergent iterative hard 
% thresholding for nonnegative sparsity optimization. Pacific Journal of 
% Optimization, 13(2), 325-353.
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!! 
% ========================================================================= 

warning off;

if nargin<3; error('Imputs are not enough!\n'); end
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

if nargin<4; pars = struct([]); end
if isfield(pars,'disp');  disp   = pars.disp; else; disp = 1;        end
if isfield(pars,'maxit'); maxit  = pars.maxit;  else; maxit  = 5e3;      end
if isfield(pars,'tol');   tol    = pars.tol;    else; tol = 1e-6*sqrt(n);end  
if isfield(pars,'neg');   neg    = pars.neg;    else; neg = 0;           end  

if isstruct(data);  data.n = n; end
func   = @(x)compressed_sensing(x,data); 


t0     = tic;
sigma0 = 1e-4;
x      = zeros(n,1);
xo     = zeros(n,1);

% main body
if disp 
    fprintf(' \n Start to run the sover -- IIHT \n'); 
    fprintf(' -------------------------------------\n');
    fprintf(' Iter          ObjVal         CPUTime \n'); 
    fprintf(' -------------------------------------\n');
end
[f,g]    = func(x);
scale    = (max(f,norm(g))>n); 
scal     = n*(scale==1)+(scale==0); 
fs       = f/scal;  
gs       = g/scal;  
for iter = 1:maxit     
    
    x_old  = x;
      
    % Line search for setp size alpha
    fx_old = fs;
    alpha  = sqrt(iter);
    for j  = 1:15
        tp = x_old-alpha*gs;
        if neg
            tp     = max(0, tp);   
            [mx,T] = maxk(tp,s);
        else
        	[mx,T] = maxk(tp,s,'ComparisonMethod','abs');    
        end
        x      = xo; 
        x(T)   = mx;
        fs     = func(x)/scal; 
        if (fs < fx_old-.5*sigma0*sum((x-x_old).^2)); break; end
        alpha  = alpha/2;        
    end
 
    [f,g]  = func(x);
    fs     = f/scal;  
    gs     = g/scal;  
    
    % Stop criteria 
	residual = scal*norm(gs(T))/max(1,norm(mx)); 
    if disp && mod(iter,1)==0 
       fprintf('%4d          %5.2e      %6.3fsec\n',iter, fs*scal,toc(t0));
    end
 
	if residual<tol || abs(fs-fx_old)<1e-12*(1+abs(fs))  
       break; 
    end  

end

out.sol   = x;
out.obj   = fs*scal;
out.iter  = iter;
out.time  = toc(t0);  
out.sp    = nnz(x);
out.error = norm(g)^2;
end


%--------------------------------------------------------------------------
function [out1,out2] = compressed_sensing(x,data)    
% data is a structure containing 
%    (data.A data.At data.b) 
% where data.At = data.A'    
    if  isa(data.A, 'function_handle')  
        Axb  = data.A(x)-data.b;
        out1 = sum(Axb.*Axb)/2;        % objective function 
        if  nargout == 2 
            out2 = data.At(Axb);       % gradien     
        end
    else
        Tx  = find(x);
        if  isempty(Tx)
            Axb  = - data.b;
        else
            Axb  = data.A(:,Tx)*x(Tx)-data.b;
        end
        out1 = sum(Axb.*Axb)/2;         % objective function 
        if  nargout == 2 
            out2 = (Axb'*data.A)';      % gradien      
        end
    end
end



