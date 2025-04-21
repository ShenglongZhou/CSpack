function Out = NHTP(data,n,s,pars)

% This code aims at solving the sparsity constrained CS:
%
%         min_{x\in R^n} 0.5||Ax-b||^2,  s.t.  \|x\|_0<=s
%
% where s is the given sparsity, which is << n.  
%       A\in\R{m by n} the measurement matrix
%       b\in\R{m by 1} the observation vector 
%=========================================================================
% Inputs:
%     data    : A structure (required)
%               (data.A, data.b) if A is a matrix 
%               (data.A, data.b, data.At) if A is a function handle
%               i.e., Ax = data.A(x); A'y = data.At(y); 
%     n       : Dimension of the solution x, (required)
%     s       : Sparsity level of x, an integer between 1 and n-1, (required)           
%     pars:     Parameters are all OPTIONAL
%               pars.x0      --  Starting point of x, pars.x0=zeros(n,1) (default)
%               pars.eta     --  A positive parameter, a default one is given related to inputs  
%               pars.disp    --  Display or not results for each iteration (default, 1)
%               pars.draw    --  A graph will be drawn or not (default,1) 
%               pars.maxit   --  Maximum number of iterations (default,2000) 
%               pars.tol     --  Tolerance of the halting condition (default,1e-6)
%               pars.obj     --  The provided objective (default 1e-20)
%                                Useful for noisy case.
% Outputs:
%     Out.sol:           The sparse solution x
%     Out.sp:            Sparsity level of Out.sol
%     Out.time           CPU time
%     Out.iter:          Number of iterations
%     Out.obj:           Objective function value at Out.sol 
%==========================================================================
% This code is programmed based on the algorithm proposed in 
% "S. Zhou, N. Xiu and H. Qi, Global and Quadratic Convergence of Newton 
% Hard-Thresholding Pursuit, Journal of Machine Learning Research, 2021."
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!! ! 
% ========================================================================= 

warning off;
t0  = tic;
if  nargin<3
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

if nargin < 4; pars = struct([]);  end 
if isfield(pars,'disp');   disp  = pars.disp;  else; disp  = 1;         end
if isfield(pars,'maxit');  itmax = pars.maxit; else; itmax = 2000;      end
if isfield(pars,'tol');    tol   = pars.tol;   else; tol   = 1e-6;      end  
if isfield(pars,'x0');     x0    = pars.x0;    else; x0    = zeros(n,1);end 
if isfield(pars,'obj');    tolF  = pars.obj;   else; tolF  = 1e-20;     end

if isstruct(data);  data.n = n; end
func           = @(x,T1,T2)CS(x,T1,T2,data); 
[x0,obj,g,eta] = getparameters(n,s,x0,func,pars);  

x       = x0;
beta    = 0.5;
sigma   = 5e-5;
delta   = 1e-10;
T0      = [];
Error   = zeros(1,itmax);
Obj     = zeros(1,itmax);
FNorm   = @(x)norm(x)^2;
xo      = zeros(n,1);

 
if  disp 
    fprintf(' \n Start to run the solver -- NHTP \n');
    fprintf(' -------------------------------------\n');
    fprintf(' Iter          ObjVal         CPUTime \n'); 
    fprintf(' -------------------------------------\n');
end
 

% Initial check for the starting point
if  FNorm(g)<1e-20 && nnz(x)<=s
    fprintf(' Starting point is a good solution. Stop NHTP\n'); 
    Out.sol  = x;
    Out.obj  = obj;
    Out.time = toc(t0);
    return;
end

if  max(isnan(g))
    x0      = zeros(n,1);
    rind    = randi(n);
    x0(rind)= rand;
    [obj,g] = func(x0,[],[]);
end
 
% The main body  
for iter = 1:itmax
     
    xtg   = x0-eta*g;
    [~,T] = maxk(abs(xtg),s); 
    TTc   = setdiff(T0,T);
    flag  = isempty(TTc);    
    gT    = g(T);
    
    % Calculate the error for stopping criteria   
    xtaus       = max(0,max(abs(g))-min(abs(x(T)))/eta);
    if flag
    FxT         = sqrt(FNorm(gT));
    Error(iter) = xtaus + FxT;
    else
    FxT         = sqrt(FNorm(gT)+ abs(FNorm(x)-FNorm(x(T))) );
    Error(iter) = xtaus + FxT;    
    end
     
    if disp
       fprintf('%4d          %5.2e      %6.3fsec\n',iter,obj,toc(t0)); 
    end
             
    % Stopping criteria
    if Error(iter)<tol || obj <= tolF; break;  end
    
    
    % update next iterate
    if  iter   == 1 || flag           % update next iterate if T==supp(x^k)     
        H       =  func(x0,T,[]); 
        if ~isa(H,'function_handle')
            d   = -H\gT;
        else
            cgit  = min(50,10*iter);
            cgtol = max(1e-6/iter,1e-20);
            d     = my_cg(H,-gT,cgtol,cgit,zeros(s,1)); 
        end
        dg      = sum(d.*gT);
        ngT     = FNorm(gT);
        if dg   > max(-delta*FNorm(d), -ngT) || isnan(dg) 
        d       = -gT; 
        dg      = ngT; 
        end
    else                              % update next iterate if T~=supp(x^k) 
        [H,D]   = func(x0,T,TTc);
        
        if isa(D,'function_handle')
           Dx   = D(x0(TTc));
        else
           Dx   = D*x0(TTc);
        end
        
        if ~isa(H,'function_handle')
            d   = H\( Dx-gT);
        else
            cgit  = min(50,10*iter);
            cgtol = max(1e-6/iter,1e-20);
            d   = my_cg(H,Dx-gT,cgtol,cgit,zeros(s,1)); 
        end
        
        Fnz     = FNorm(x(TTc))/4/eta;
        dgT     = sum(d.*gT);
        dg      = dgT-sum(x0(TTc).*g(TTc));
        
        delta0  = delta;
        if Fnz  > 1e-4; delta0 = 1e-4; end
        ngT     = FNorm(gT);
        if dgT  > max(-delta0*FNorm(d)+Fnz, -ngT) || isnan(dg) 
        d       = -gT; 
        dg      = ngT; 
        end            
    end
    
    alpha    = 1; 
    x        = xo;    
    obj0     = obj;        
    Obj(iter)= obj;
    
    % Amijio line search
    for i      = 1:6
        x(T)   = x0(T) + alpha*d;
        obj    = func(x,[],[]);
        if obj < obj0  + alpha*sigma*dg; break; end        
        alpha  = beta*alpha;
    end
    
    % Hard Thresholding Pursuit if the obj increases
    fhtp    = 0;
    if obj  > obj0 
       x(T) = xtg(T); 
       obj  = func(x,[],[]); 
       fhtp = 1;
    end
    
    % Stopping criteria
    flag1   = (abs(obj-obj0)<1e-6*(1+abs(obj)) && fhtp); 
    flag2   = (abs(obj-obj0)<1e-10*(1+abs(obj))&& Error(iter)<1e-2);
    if  iter>10 &&  (flag1 || flag2)      
        if obj > obj0
           iter    = iter-1; 
           x       = x0; 
           T       = T0; 
        end   
        break;
     end 
 
    T0      = T; 
    x0      = x; 
    [obj,g] = func(x,[],[]);
    
    % Update eta
    if  mod(iter,50)==0  
        if Error(iter)>1/iter^2  
        if iter<1500; eta = eta/1.05; 
        else;         eta = eta/1.5; 
        end     
        else;         eta = eta*1.25;   
        end
    end     
end

% results output
time        = toc(t0);
Out.sp      = nnz(x);
Out.time    = time;
Out.iter    = iter;
Out.sol     = x;
Out.obj     = obj;  
Out.error   = FNorm(g);
end

% initialize parameters ---------------------------------------------------
function [x0,obj,g,eta]=getparameters(n,s,x0,func,pars)

    if isfield(pars,'x0') && norm(x0)>0
       [obj0,g0] = func(zeros(n,1),[],[]);  
       [obj,g]   = func(pars.x0,[],[]); 
       if obj0   < obj/10
          x0     =  zeros(n,1); 
          obj    = obj0;  
          g      = g0; 
       else  
          ns0    = nnz(pars.x0);
          if ns0==s
          [~,T]    = maxk(pars.x0,s,'ComparisonMethod','abs'); 
          x0       = pars.x0;  
          pars.eta = min(abs(x0(T)))/(1+max(abs(g(setdiff(1:n, T)))));   
          elseif ns0<s
          x0        = pars.x0;  
          pars.eta  = max(x0(x0>0.1))/(1+max(abs(g)));   
          else 
          [~,T]     = maxk(pars.x0,s,'ComparisonMethod','abs'); 
          x0        = zeros(n,1);
          x0(T)     = pars.x0(T);  
          pars.eta  = max(x0(x0>0.1))/(1+max(abs(g)));  
          end
          
          if isempty(pars.eta) 
          pars.eta  = max(abs(x0))/(1+max(abs(g))); 
          end
          
       end
    else
        [obj,g]  = func(x0,[],[]); 
    end
 
    
    if isfield(pars,'eta')      
        eta  = pars.eta;       
    else % set a proper parameter eta
        [~,g1] = func(ones(n,1),[],[]) ;
        abg1   = abs(g1);
        T      = find(abg1>1e-8);
        maxe   = sum(1./(abg1(T)+eps))/nnz(T);
        if  isempty(T) 
            eta    = 10*(1+s/n)/min(10, log(n));
        else
            if maxe>2
            eta  = (log2(1+maxe)/log2(maxe))*exp((s/n)^(1/3));
            elseif maxe<1
            eta  = (log2(1+ maxe))*(n/s)^(1/2);    
            else
            eta  = (log2(1+ maxe))*exp((s/n)^(1/3));
            end     
        end
    end 
    
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
    if ~isa(fx,'function_handle'); fx = @(v)fx*v; end
    r = b;
    if nnz(x)>0; r = b - fx(x);  end
    e = norm(r,'fro')^2;
    t = e;
    p = r;
    for i = 1:cgit  
        if e < cgtol*t; break; end
        w  = fx(p);
        pw = p.*w;
        a  = e/sum(pw(:));
        x  = x + a * p;
        r  = r - a * w;
        e0 = e;
        e  = norm(r,'fro')^2;
        p  = r + (e/e0)*p;
    end 
end
