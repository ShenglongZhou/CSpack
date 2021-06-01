function out = NL0R(data,n,pars)
% This code aims at solving the L0 regularized optimization with form
% 
%         min_{x\in R^n} 0.5||Ax-b||^2 + \lambda \|x\|_0 
% 
% where \lambda is updated iteratively.
%       A\in\R{m by n} the measurement matrix
%       b\in\R{m by 1} the observation vector 
% =========================================================================
% Inputs:
%     data:     A structure (required)
%               (data.A, data.b) if A is a matrix 
%               (data.A, data.b, data.At) if A is a function handle
%               i.e., Ax = data.A(x); A'y = data.At(y); 
%     n:        Dimension of the solution x, (required)             
%     pars:     Parameters are all OPTIONAL
%               pars.x0      --  Starting point of x (default, zeros(n,1))
%               pars.tau     --  A positive scalar (default,(n<=1e3)+(n>1e3)/2)
%               pars.lam     --  An initial penalty parameter (default, maxlam/2)
%               pars.rate    --  A positive scalar to adjust lam, (default, rate0) 
%               pars.disp    --  Display or not results for each iteration (default, 1) 
%               pars.maxit   --  Maximum number of iterations, (default,2000) 
%               pars.tol     --  Tolerance of the halting condition, (default,1e-6)
%               pars.obj     --  A predefined lower bound of f(x), (default,1e-20)
% Outputs:
%     out.sol:           The sparse solution x
%     out.sp:            Sparsity level of out.sol 
%     out.time           CPU time
%     out.iter:          Number of iterations
%     out.obj:           Objective function value at out.sol 
% ========================================================================= 
% This code is programmed based on the algorithm proposed in 
% S. Zhou, L. Pan and N. Xiu, Numerical Algorithms, 2020, 
% Newton Method for l_0 Regularized Optimization.
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!! 
% ========================================================================= 

warning off;

t0 = tic;
if nargin<2
   fprintf(' No enough inputs. No problems will be solverd!'); return;
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

if isstruct(data);  data.n = n; end
func   = @(x,T1,T2)CS(x,T1,T2,data);

if nargin<3; pars=[]; end    
    
rate0 = (n<=1e3)*0.5+(n>1e3)/exp(3/log10(n));
tau0  = (n<=1e3)+(n>1e3)/2; 
if isfield(pars,'x0');    x0    = pars.x0;    else; x0 = zeros(n,1);end
if isfield(pars,'tau');   tau   = pars.tau;   else; tau   = tau0;   end
if isfield(pars,'rate');  rate  = pars.rate;  else; rate  = rate0;  end
    
if isfield(pars,'disp');  disp  = pars.disp;  else; disp  = 1;      end
if isfield(pars,'maxit'); itmax = pars.maxit; else; itmax = 2000;   end
if isfield(pars,'obj');   pobj  = pars.obj;   else; pobj  = 1e-20;  end 
if isfield(pars,'tol');   tol   = pars.tol;   else; tol   = 1e-10;  end 


x       = x0;
I       = 1:n;
Err     = zeros(1,itmax);
Obj     = zeros(1,itmax);
Nzx     = zeros(1,itmax);
FNorm   = @(x)norm(x)^2;

if disp 
   fprintf(' Start to run the solver -- NL0R \n'); 
   fprintf(' ----------------------------------------------\n');
   fprintf(' Iter       ObjVal       CPUTime     Sparsity\n'); 
   fprintf(' ----------------------------------------------\n');
end

% Initial check for the starting point
[obj,g] = func(x,[],[]);
if FNorm(g)==0 
   fprintf('Starting point is a good stationary point, stop !!!\n'); 
   out.sol = x;
   out.obj = obj;
   return;
else   
   if  isfield(pars,'lam') 
       lam    = pars.lam; 
       maxlam = lam*2;
   else 
       maxlam = max(abs(g))^2*tau/2;
       lam    = maxlam/2; 
   end 

end

if  max(isnan(g))
    x       = zeros(n,1);
    rind    = randi(n);
    x(rind) = rand;
    [obj,g] = func(x,[],[]);
end

pcgit   = 5;
pcgtol  = 1e-5;
beta    = 0.5;
sigma   = 5e-5;
delta   = 1e-10;
T0      = [];  
mark    = 0;
nx      = 0;

% The main body  
for iter  = 1:itmax
    x0    = x; 
    xtg   = x0-tau*g; 
    T     = find(abs(xtg)>=sqrt(2*tau*lam)); 
    nT    = nnz(T);
    TTc   = setdiff(T0,T);
    flag  = isempty(TTc);    

    % Calculate the error for stopping criteria 
    FxT       = sqrt(FNorm(g(T))+FNorm(x(TTc)));
    Err(iter) = FxT/sqrt(n); 
    Nzx(iter) = nx;
    if  disp  
        fprintf('%4d       %6.2e     %6.3fsec     %6d\n',iter, obj, toc(t0), nx); 
    end
    
    % Stopping criteria   
    stop1  = (Err(iter)<tol); 
    stop2  = (iter>1 && abs(obj-obj0)<1e-10*(1+obj));
    stop3  = (nx==nT);
    if (stop1 && stop2 && stop3 && flag) 
        mark = mark + 1; 
    end
    
    stop4 =  obj  < pobj;
    stop5 =  iter > 9 && std(Nzx(iter-9:iter))<= 0 &&...
             std(Err(iter-9:iter))^2 <= min(Err(iter-9:iter)) &&...
             std(Obj(iter-9:iter))^2 <= min(Obj(iter-9:iter-1));
    if mark==5 || stop4 || stop5, break;   end
   
    % update next iterate
    if  iter   == 1 || flag    % two consective iterates have same supports
        H       = func(x0,T,[]);     
        if isa(H,'function_handle')
           d    = my_cg(H,-g(T),pcgtol,pcgit,zeros(nT,1)); 
        else 
           d    = -H\g(T);  
        end
       
        dg     = sum(d.*g(T));
        ngT    = FNorm(g(T));
        if dg  > max(-delta*FNorm(d), -ngT) || isnan(dg) 
        d      = -g(T); 
        dg     = ngT; 
        end
    else                  % two consective iterates have different supports                       
        [H,D]  = func(x0,T,TTc);             
        if  isa(H,'function_handle')
            d  = my_cg(H, D(x0(TTc))-g(T),pcgtol,pcgit,zeros(nT,1));  
        else
            d  = H\( D(x0(TTc)) - g(T)); 
        end
         
        Fnz    = FNorm(x(TTc))/4/tau;
        dgT    = sum(d.*g(T));
        dg     = dgT-sum(x0(TTc).*g(TTc));
        
        delta0 = delta;
        if Fnz > 1e-4; delta0 = 1e-4; end
 
        ngT    = FNorm(g(T));
        if dgT > max(-delta0*FNorm(d)+Fnz, -ngT) || isnan(dg) 
           d   = -g(T); 
           dg  = ngT; 
        end            
    end
    
    % Armijo line search
    alpha    = 1; 
    x        = zeros(n,1);    
    obj0     = obj;             
    for i      = 1:6
        x(T)   = x0(T) + alpha*d;
        obj    =  func(x,[],[]);
        if obj < obj0  + alpha*sigma*dg; break; end        
        alpha  = beta*alpha;
    end
    

    T0       = T; 
    [obj,g]  = func(x,[],[]);
    Obj(iter)= obj; 
    
%   Update tau    
    if  mod(iter,10)==0  
        OBJ = Obj(iter-9:iter);
        if Err(iter)>1/iter^2 || sum(OBJ(2:end)>1.5*OBJ(1:end-1))>=2 
            if iter<1500; tau = tau/1.25; 
            else;         tau = tau/1.5; 
            end     
        else          
            tau = tau*1.25;   
        end
    end 
    
%   Update lambda    
    nx  = nnz(x); 
    if  iter>5 && (nx > 2*max(Nzx(1:iter-1))) && Err(iter)<1e-2
        rate0   = 2/rate;   
        x       = x0;
        nx      = nnz(x0); 
        nx0     = Nzx(iter-1);  
        [obj,g] = func(x,[],[]);
        rate    = 1.1;
    else  
        rate0   = rate;
    end
       
    if exist('nx0') && nx < nx0
       rate0 = 1;   
    end
 
    if mod(iter,1)==0 
       lam  = min(maxlam,lam*(2*(nx>=0.1*n)+rate0));
    end
    

end

%Results output ------------------------------------------------- 

iter        = iter-1;
x           = SparseApprox(x,n);
[obj,g]     = func(x,[],[]);
time        = toc(t0);
out.sp      = nnz(x);  
out.time    = time;
out.iter    = iter;
out.sol     = x;
out.obj     = obj;   


if disp 
   fprintf(' ----------------------------------------------\n');
   normgrad    = FNorm(g);
   if normgrad < 1e-10
      fprintf(' A global optimal solution might be found\n');
      fprintf(' because of ||gradient|| = %5.2e!\n',normgrad); 
      fprintf(' ----------------------------------------------\n');
   end
end

end

% get the sparse approximation of x ---------------------------------------
function sx = SparseApprox(x0,n)
x       = abs(x0);
T       = find(x);
[sx,id] = sort(x(T),'descend'); 
y       = 0;
nx      = sum(x(T));
nT      = nnz(T);
t       = zeros(nT-1,1);
stop    = 0;
for i   = 1:nT
    if y > 0.99995*nx && stop; break; end
    y    = y + sx(i); 
    if i < nT
    t(i) = sx(i)/sx(i+1);
    stop = (t(i)>1e3);
    end  
end
 
if  i  < nT
    j  = find(t==max(t));  
    i  = j(1);
else
    i  = nT;
end
 
sx = zeros(n,1);
sx(T(id(1:i))) = x0(T(id(1:i)));
end

% define functions --------------------------------------------------------
function [out1,out2] = CS(x,T1,T2,data)

if ~isa(data.A, 'function_handle') % A is a matrix 
    if  isempty(T1) && isempty(T2) 
        if  nnz(x) >= 0.8*length(x)
            Axb     = data.A*x-data.b;
        else
            Tx      = find(x); 
            Axb     = data.A(:,Tx)*x(Tx)-data.b;
        end
            out1    = (Axb'*Axb)/2;               % objective function value of f
        if  nargout == 2
            out2    = (Axb'*data.A)';                % gradien of f
        end
    else        
            AT = data.A(:,T1); 
        if  length(T1)<3000
            out1 = AT'*AT;                        %subHessian containing T1 rows and T1 columns
        else
            out1 = @(v)( (AT*v)'*AT )';      
        end       
        if  nargout == 2
            out2 = @(v)( (data.A(:,T2)*v)'*AT )'; %subHessian containing T1 rows and T2 columns
        end       
    end
else  % A is a function handle A*x=A(x)  
    if ~isfield(data,'At') 
        disp('The transpose-data.At-is missing'); return; 
    end
    if ~isfield(data,'n')  
        disp('The dimension-data.n-is missing');  return;  
    end   
    if  isempty(T1) && isempty(T2)  
        Axb  = data.A(x)-data.b;
        out1 = (Axb'*Axb)/2;              % objective function value of f
        if  nargout>1 
            out2 = data.At(Axb);          % gradien of f
        end
    else
        func = fgH(data);    
        out1 = @(v)func(v,T1,T1);         % subHessian containing T1 rows and T1 columns
        if  nargout>1
            out2 = @(v)func(v,T1,T2);     % subHessian containing T1 rows and T1 columns
        end  
        
    end
end

end

function Hess = fgH(data)
    suppz     = @(z,t)supp(data.n,z,t);
    sub       = @(z,t)z(t,:);
    Hess      = @(z,t1,t2)(sub( data.At( data.A(suppz(z,t2))),t1)); 
end

function z = supp(n,x,T)
    z      = zeros(n,1);
    z(T)   = x;
end
% conjugate gradient-------------------------------------------------------
function x = my_cg(fx,b,cgtol,cgit,x)
    r = b;
    e = sum(r.*r);
    t = e;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        if  i == 1  
            p = r;
        else
            p = r + (e/e0)*p;
        end  
        if  isa(fx,'function_handle')
            w  = fx(p);
        else
            w  = fx*p;
        end
        a  = e/sum(p.*w);
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = sum(r.*r);
    end  
end




