function Out = PSNP(data,n,lambda,pars)
% This code aims at solving the Lq norm regularized optimization with form
%
%         min_{x\in R^n} 0.5||Ax-b||^2 + \lambda \lambda \|x\|_q^q,
%
% where \lambda>0, q \in [0,1)
%       A\in\R{m by n} the measurement matrix
%       b\in\R{m by 1} the observation vector  
%--------------------------------------------------------------------------
% Inputs:
%     func:  A function handle defines (objective,gradient,sub-Hessain) of F (required)
%     n   :  Dimension of the solution x                                     (required)
%     lambda : Penalty parameter                                             (required) 
%     pars:  Parameters are all OPTIONAL
%            pars.x0      --  Starting point of x,   pars.x0 = zeros(n,1)  (default,   0)
%            pars.q       --  Decide the Lq norm                           (default, 0.5)
%            pars.cond    --  Conditional or unconditional Newton method   (default,   1)  
%            pars.show    --  Display results or not for each iteration    (default,   1)
%            pars.maxit   --  Maximum number of iterations                 (default, 2e3) 
%            pars.tol     --  Tolerance of the halting condition           (default,1e-6)
%
% Outputs:
%     Out.sol :   The sparse solution x
%     Out.time:   CPU time
%     Out.iter:   Number of iterations
%     Out.obj :   Objective function value at Out.sol 
%--------------------------------------------------------------------------
% This code is programmed based on the algorithm proposed in 
% "S. Zhou, X. Xiu, Y. Wang, and D. Peng, 
%  Revisiting Lq(0<=q<1) norm regualrized optimization, 2023."
% Send your comments and suggestions to <<< slzhou2021@163.com >>> 
% Warning: Accuracy may not be guaranteed !!!!! 
%--------------------------------------------------------------------------

warning off;
t0  = tic;
if  nargin < 3
    disp(' No enough inputs. No problems will be solverd!'); return;
end
if nargin < 4; pars = [];  end 

[sig1,sig2,q,q1,q2,lamq,lamq1,alpha0,cg_tol,cg_it,x0,tol,maxit,newton,...
         show,cond,rate,i0] = setparameters(n, lambda, pars);
Fnorm    = @(var)norm(var,'fro')^2; 
Funcs    = @(xT,T,key)funCS(xT,T,n,key, data);
ProxLq   = @(x,t)ProxmaLq(x,t,q);
Qnorm    = @(absvar)sum(absvar.^q); 
GQnorm   = @(var,absvar)(sign(var).*absvar.^q1); 
HQnorm   = @(absvar)(absvar.^q2); 
Count    = @(it)((it<=10)+10*(it>10)+90*(it>100));
Error    = zeros(maxit,1);
x        = x0; 
T        = find(x~=0); 
sT       = nnz(T);
w        = x(T);
if sT       > 0
    absw    = abs(w);
    Obj     = Funcs(w,T,'f') + lambda*Qnorm(absw); 
    grad    = Funcs(w,T,'g');
    grad(T) = grad(T) + lamq*GQnorm(w,absw);
else
    Obj  = Funcs(x,[],'f');
    grad = Funcs(x,[],'g'); 
end
 
if  show 
    if  cond 
        fprintf(' \nStart to run the solver -- PCSNP\n'); 
    else
        fprintf(' \n Start to run the solver -- PSNP\n'); 
    end
    fprintf(' -------------------------------------\n');
    fprintf(' Iter          ObjVal         CPUTime \n'); 
    fprintf(' -------------------------------------\n'); 
    fprintf('%4d          %5.2e      %6.3fsec\n',  0, Obj, toc(t0));
end

% The main body
for iter = 1:maxit

    alpha       = alpha0; 
    Told        = T;
    for i       = 1:i0  
        [w,T]   = ProxLq(x-alpha*grad,alpha*lambda);        
        absw    = abs(w);  
        fx      = Funcs(w,T,'f');
        Objw    = fx+lambda*Qnorm(absw);  
        if isempty(T) || Objw < Obj - sig1*Fnorm(w-x(T))
           break; 
        end 
        alpha   = alpha*rate;
    end

    if isempty(T)
       alpha   = alpha/rate; 
       [w,T]   = ProxLq(x-alpha*grad,alpha*lambda); 
       absw    = abs(w);  
       fx      = Funcs(w,T,'f');
       Objw    = fx+lambda*Qnorm(absw);
    end
       
    x       = zeros(n,1);
    x(T)    = w;
    Objold  = Obj;
    Obj     = Objw; 
    sTold   = sT; 
    sT      = nnz(T); 
    ident   = 0;
    if sT   == sTold 
       ident = nnz(T-Told)==0;   
    end     
 
    if cond
       switchon = (ident ==1);
    else
       switchon = (sT < 0.25*n || ident ==1) && (i~=i0 || iter>=5);
    end
 
    if  newton && sT>0 && switchon
        [grad,Hess] = Funcs(w,T,'gh'); 
        gradT       = grad(T);
        if q        > 0
            gradT   = gradT + lamq*GQnorm(w,absw);
            dw      = lamq1*HQnorm(absw);
            if isa(Hess, 'function_handle') 
                Hess = @(v)(Hess(v) + dw.*v);
            else
                Hess(1:(sT+1):end) = Hess(1:(sT+1):end) + dw';
            end
        end
        if  isa(Hess, 'function_handle')    
            if  iter   > 10  
                cg_tol = max(cg_tol/10,1e-15*sT);
                cg_it  = min(cg_it+5,25);
            end    
            d    = my_cg(Hess,gradT,cg_tol,cg_it,zeros(sT,1));  
        else 
            d    = Hess\gradT; 
        end
        
        beta     = 1;
        Fd       = Fnorm(d);   
        for j    = 1 : 5
            v    = w - beta* d;
            absv = abs(v);
            fx   = Funcs(w,T,'f');
            Objv = fx + lambda*Qnorm(absv);          
            if  Objv <= Obj - sig2*beta^2*Fd 
                x(T) = v; 
                w    = v; 
                absw = absv;
                Obj  = Objv; 
                break;
            end 
            beta = beta * 0.25;
        end        
    end
    
    grad        = Funcs(w,T,'g'); 
    gradT       = grad(T);
    if q        > 0
        gradT   = gradT + lamq*GQnorm(w,absw);  
    end
    if  iter>1 && isempty(T) 
        ErrGradT = 1e10; 
        lambda   = lambda/1.5; 
        lamq     = lambda*q;
        lamq1    = lamq*q1;
    else
        ErrGradT = norm(gradT,'inf'); 
    end
    ErrObj      = abs(Obj-Objold)/(1+abs(Obj));       
    Error(iter) = ErrGradT/sqrt(n);
    
    if  show   
        fprintf('%4d          %5.2e      %6.3fsec\n',  iter, fx, toc(t0)); 
    end
             
    % Stopping criteria
     if  ( (n>5e4 && isa(data.A, 'function_handle')) || ident) && max(Error(iter),ErrObj) <tol
         break;  
     end
end

Out.time    = toc(t0);
Out.iter    = iter;
Out.sol     = x;
Out.obj     = Obj;  
Out.error   = Fnorm(grad); 
end

% Set up parameters -------------------------------------------------------
function [sig1,sig2,q,q1,q2,lamq,lamq1,alpha0,cg_tol,cg_it,x0,tol,maxit,...
          newton,show,cond,rate,i0] = setparameters(n, lambda, pars)

if isfield(pars,'x0');     x0     = pars.x0;     else; x0 = zeros(n,1); end 
if isfield(pars,'q');      q      = pars.q;      else; q      = 0.5;    end 
if isfield(pars,'tol');    tol    = pars.tol;    else; tol    = 1e-6;   end  
if isfield(pars,'maxit');  maxit  = pars.maxit;  else; maxit  = 1e4;    end
if isfield(pars,'newton'); newton = pars.newton; else; newton = 1;      end 
if isfield(pars,'show');   show   = pars.show;   else; show   = 1;      end 
if isfield(pars,'cond');   cond   = pars.cond;   else; cond   = 1;      end 

sig1  = 1e-6; 
sig2  = 1e-10;
q1    = q-1;
q2    = q-2;
lamq  = lambda*q;
lamq1 = lambda*q*(q-1);

alpha0 = 1-q/2;
cg_tol = 1e-8; 
cg_it  = 10;
rate   = 0.5; 
i0     = 6;
end

function [out1,out2] = funCS(xT,T,n,key, data)

    mark = isa(data.A, 'function_handle'); 
    if mark
        if ~isfield(data,'At') 
            disp('The transpose-data.At-is missing'); return; 
        end
        Atb = @(v)data.At(v);
    else
        Atb = @(v)(v'*data.A)';
    end

    if  isempty(T) 
        Axb = -data.b; 
        if isequal(key, 'f')
            out1 = norm(Axb,'fro')^2/2;
        else
            out1 = Atb(Axb);    
            if isequal(key,'gh'); out2 = []; end
        end            
    else    
        if ~mark         
            AT  = data.A(:,T);
            Axb = AT*xT-data.b; 
        else
            Axb = data.A(supp(n,xT,T))-data.b; 
        end

        if isequal(key, 'f')
            out1 = norm(Axb,'fro')^2/2;
        else
            out1 = Atb(Axb);
            if  isequal(key,'gh')  
                if ~mark
                    if nnz(T) < 500 && size(data.A,1)<500
                       out2 = AT'*AT;  
                    else
                       out2 = @(v)((AT*v)'*AT)' ;      
                    end 
                else
                    out2 = @(v)sub( Atb( data.A(supp(n,v,T))),T);
                end
            end
        end     
    end
end

function z = supp(n,x,T)
    z      = zeros(n,1);
    z(T)   = x;
end

function subz = sub(z,T)
         subz = z(T,:);
end


% Proximal of Lq norm------------------------------------------------------
function [px,T] = ProxmaLq(a,lam,q)

% solving problem   xopt = argmin_x 0.5*||x-a||^2 + lam*||x||_q^q

% a:    a vector
% lam:  a positive scalar
% q:    a scalar in [0,1)

% px:   px = xopt(T);
% T :   the support set of xopt

switch q
    case 0
         t     = sqrt(2*lam);
         T     = find(abs(a)>t);  
         px    = a(T);        
    case 1/2 
         t     = (3/2)*lam^(2/3);
         T     = find(abs(a) > t);
         aT    = a(T);
         phi   = acos( (lam/4)*(3./abs(aT)).^(3/2) );
         px    = (4/3)*aT.*( cos( (pi-phi)/3) ).^2;
    case 2/3
         t     = 2*(2*lam/3)^(3/4); 
         T     = find( abs(a) >  t );  
         aT    = a(T);       
         tmp1  = aT.^2/2; 
         tmp2  = sqrt( tmp1.^2 - (8*lam/9)^3 );  
         phi   = (tmp1+tmp2).^(1/3)+(tmp1-tmp2).^(1/3);
         px    = sign(aT)/8.*( sqrt(phi)+sqrt(2*abs(aT)./sqrt(phi)-phi) ).^3; 
    otherwise
         [px,T] = NewtonLq(a,lam,q);
end

end
% Newton method for proximal of Lq norm------------------------------------
function [w,T] = NewtonLq(a,alam,q)

    thresh = (2-q)*alam^(1/(2-q))*(2*(1-q))^((1-q)/(q-2));
    T      = find(abs(a)>thresh); 

    if ~isempty(T)
        zT     = a(T);
        w      = zT;
        maxit  = 1e2;
        q1     = q-1;
        q2     = q-2;
        lamq   = alam*q;
        lamq1  = lamq*q1;

        gradLq = @(u,v)(u - zT + lamq*sign(u).*v.^q1);
        hessLq = @(v)(1+lamq1*v.^q2);
        func   = @(u,v)(norm(u-zT)^2/2+alam*sum(v.^q));

        absw   = abs(w);
        fx0    = func(w,absw); 

        for iter  = 1:maxit
            g     = gradLq(w,absw);
            d     = -g./hessLq(absw); 
            alpha = 1;  
            w0    = w;
            for i    = 1:10
                w    =  w0 + alpha*d;  
                absw = abs(w);
                fx   = func(w,absw);
                if  fx < fx0 - 1e-4*norm(w-w0)^2
                   break; 
                end 
                alpha   = alpha*0.5;
            end
            if  norm(g) < 1e-8; break; end
        end
    else
        w = [];
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
