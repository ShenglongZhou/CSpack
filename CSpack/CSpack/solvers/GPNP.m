function out = GPNP(data,n,s,pars)
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
%               pars.disp    --  Display or not results for each iteration (default, 1)
%               pars.maxit   --  Maximum number of iterations (default,2000) 
%               pars.tol     --  Tolerance of the halting condition (default,1e-6)
%               pars.obj   --  The provided objective (default 1e-20)
%                              Useful for noisy case.
% Outputs:
%     Out.sol:           The sparse solution x
%     Out.sp:            Sparsity level of Out.sol
%     Out.time           CPU time
%     Out.iter:          Number of iterations
%     Out.obj:           Objective function value at Out.sol 
% =========================================================================  
% This code is programmed based on the algorithm proposed in 
% S. Zhou,  2022, Applied and Computational Harmonic Analysis,
% Gradient projection newton pursuit for sparsity constrained optimization
%%%%%%%    Send your comments and suggestions to                     %%%%%%
%%%%%%%    slzhou2021@163.com                                        %%%%%% 
%%%%%%%    Warning: Accuracy may not be guaranteed!!!!!              %%%%%%
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

funhd     = isa(data.A,'function_handle'); 
[x0,sigma,J,flag,m,alpha0,gamma,thd,disp,tol,tolF,maxit]...
          = set_parameters(s,n,data.b,pars);

x         = x0; 
bt        = data.b';
Fnorm     = @(var)norm(var)^2;
supp      = @(x,T)support(x,n,T);

if  funhd    
    funAv  = @(v)data.A(v);
    funAtv = @(v)data.At(v);
    OBJ    = zeros(3,1);
    sub    = @(z,t)z(t,:);
    subH   = @(z,T)(sub(funAtv(funAv(supp(z,T))),T));         
else
    funAv  = @(v,T)data.A(:,T)*v; 
    if  isfield(data,'At')
        funAtv = @(v)data.At*v; 
    else
        funAtv = @(v)(v'*data.A)';
    end
    OBJ   = zeros(5,1);
end

gx        = funAtv(data.b);
Atb       = gx;   
fx        = Fnorm(data.b);
[~,Tx]    = maxk(gx,s,'ComparisonMethod','abs');
Tx        = sort(Tx);  
minobj    = zeros(maxit,1);
minobj(1) = fx;

% main body
if  disp 
    fprintf(' Start to run the solver -- GPNP \n');
    fprintf(' ------------------------------------------------\n');
    fprintf(' Iter       Error          ObjVal         CPUTime \n'); 
    fprintf(' ------------------------------------------------\n');
end
 
for iter = 1:maxit     
     
    % Line search for setp size alpha
 
    alpha  = alpha0;  
    for j  = 1:J
        [subu,Tu] = maxk(x-alpha*gx,s,'ComparisonMethod','abs');
        u         = supp(subu,Tu);  
        if funhd  
            Aub   = funAv(u)-data.b;
        else
            Aub   = funAv(subu,Tu)-data.b;
        end      
        fu        = Fnorm(Aub);  
        if fu     < fx - sigma*Fnorm(u-x); break; end
        alpha     = alpha*gamma;        
    end
 
    gx      = funAtv(Aub);
    normg   = Fnorm(gx);
    x       = u;
    fx      = fu; 
    
    % Newton step
    sT   = sort(Tu); 
    mark = nnz(sT-Tx)==0;
    Tx   = sT;
    eps  = 1e-4;
    if ( mark || normg < 1e-4  || alpha0==1 ) && s<=5e4
        if funhd  
           cgit     = min(50,5*iter);
           cgtol    = max(1e-6/10^iter,1e-20);
           subv     = my_cg(@(var)subH(var,Tu),Atb(Tu),cgtol,cgit,zeros(s,1));
        else 
           ATu      = data.A(:,Tu); 
           if  s   <= 1000 && m <= 10000
               subv = (ATu'*ATu)\(bt*ATu)'; 
               eps  = 1e-10;
           else
               if  issparse(data.A)
                   cgit = 30+(s/n>=0.05)*20+(s/n>=0.1)*20;  
               else
                   cgit = min(20,2*iter);  
               end
               subv = my_cg(@(var)((ATu*var)'*ATu)',Atb(Tu),1e-20,cgit,zeros(s,1)); 
           end           
        end 
        v           = supp(subv,Tu);
        if funhd  
            Avb     = funAv(v)-data.b;
        else
            Avb     = funAv(subv,Tu)-data.b;
        end
        fv          = Fnorm(Avb);  
        if fv      <= fu  - sigma * Fnorm(subu-subv)
           x        = v;  
           fx       = fv;
           subu     = subv;  
           gx       = funAtv(Avb); 
           normg    = Fnorm(gx); 
        end   
    end
    
    % Stop criteria  
    error     = Fnorm(gx(Tu)); 
    obj       = sqrt(fx);
    OBJ       = [OBJ(2:end); obj];
    if disp  
       fprintf('%4d       %5.2e       %5.2e      %6.3fsec\n',iter,error,fx,toc(t0)); 
    end

    maxg      = max(abs(gx));
    minx      = min(abs(subu));
    J         = 8;
    if error  < tol*1e3 && normg>1e-2 && iter < maxit-10
       J      = min(8,max(1,ceil(maxg/minx)-1));     
    end  

    if isfield(pars,'obj')  && obj <=tolF && flag
        maxit  = iter + 100*s/n; 
        flag   = 0; 
    end
 
    minobj(iter+1) = min(minobj(iter),fx);  
    if fx    < minobj(iter) 
        xmin = x;  
        fmin = fx;  
    end
    
    if iter  > thd  
       count = std(minobj(iter-thd:iter+1) )<1e-10;
    else
       count = 0; 
    end

    if  normg<tol || fx < tolF  || ...
        count  || (std(OBJ)<eps*(1+obj)) 
        if count && fmin < fx; x=xmin; fx=fmin; end
        break; 
    end  
    
end

out.sol   = x;
out.obj   = fx;
out.sp    = nnz(x);
out.iter  = iter;
out.time  = toc(t0);

if  disp
    fprintf(' ------------------------------------------------\n');
end
if  fx<1e-10 && disp
    fprintf(' A global optimal solution may be found\n');
    fprintf(' because of ||Ax-b|| = %5.3e!\n',sqrt(fx)); 
    fprintf(' ------------------------------------------------\n');
end

end

%-------------------------------------------------------------------------- 
function z = support(x,n, T)
      z    = zeros(n,1);
      z(T) = x;
end

% set parameters-------------------------------------------------------
function [x0,sigma,J,flag,m,alpha0,gamma,thd,disp,tol,tolF,maxit]=set_parameters(s,n,b,pars)
    sigma     = 1e-4; 
    J         = 1;    
    m         = length(b);
    flag      = 1;

	if  m/n   >= 1/6 && s/n <= 0.05 && n>=1e4
        alpha0 = 1; gamma = 0.1;
    else
        alpha0 = 5; gamma = 0.5;
    end
    
    if s/n  <= 0.05
       thd   = ceil(log2(2+s)*100); 
    else
       thd   = ceil(log2(2+s)*750);
    end   

    if isfield(pars,'x0');     x0   = pars.x0;    else; x0 = zeros(n,1);end 
    if isfield(pars,'disp');  disp  = pars.disp;  else; disp  = 1;      end
    if isfield(pars,'tol');   tol   = pars.tol;   else; tol   = 1e-10;  end  
    if isfield(pars,'obj');   tolF  = pars.obj;   else; tolF  = 1e-20;  end 
    if isfield(pars,'maxit')
        maxit = pars.maxit;  
    else 
        maxit = (n<=1e4)*2e4 + (n>1e4)*5e3; 
    end   
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
