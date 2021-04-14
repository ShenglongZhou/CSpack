function  Out= MIRL1(Data,n,pars)

% A solver for reweighted L1-minimization model:
%
%    min 0.5||Ax-b||^2 + mu||w.*x||_1
%
% Written by 05/05/2015, Shenglong Zhou
%
% Note: yall1 solver is taken from http://yall1.blogs.rice.edu/

% --- Inputs:
%     data --- A triple structure (data.A, data.At, data.b) (required)
%              data.A, the measurement matrix, or a function handle @(x)A(x) 
%              data.At = data.A',or a function handle @(x)At(x) 
%              data.b, the observation vector 
%     n    --- Dimension of the solution x (required)
%     pars --- a structure with fields:
%              pars.tol    -- tolerance for yall1 solver (default, 1e-4)
%              pars.rate   -- for updating the sparsity (default,1/(log(n/m)) 
%              pars.s      -- for the given sparsity level if it is known in advance 
%              pars.disp   -- display results in each iteration, (default, 1)
% --- Outputs:
%     Out ---  a structure with fields:
%              Out.sol     -- recovered solution, an n x 1  order vector 
%              Out.sp      -- Sparsity level of Out.sol
%              Out.iter    -- number of total iterations 
%              Out.time    -- total computational time 
%              Out.obj     -- Objective function value at Out.sol  
%
% This code is programmed based on the algorithm proposed in 
% "S. Zhou, N. Xiu, Y. Wang, L. Kong and H. Qi, 
% A Null-space-based weighted l1 minimization approach to compressed sensing, 
% Information and Inference: A Journal of the IMA, 5(1),76-102, 2016."
% Send your comments and suggestions to <slzhou2021@163.com> 
% Warning: Accuracy may not be guaranteed !!!!! 

if nargin<2
   error('Inputs are not enough')
elseif nargin==2
   pars=[];
end

A           = Data.A;
At          = Data.At;
b           = Data.b;
m           = length(b);                 
[itmax,rate,tol,disp,mu,i0,theta]...
            = Get_parameters(m,n,At,b,pars);     
x           = zeros(n,1);               
w           = ones(n,1);
opts_ya.tol = tol;
t_start     = tic;

if isa(A,'function_handle')
    Aya.times= @(x)A(x);
    Aya.trans= @(x)At(x);
else
    Aya = A;
end

if disp
   fprintf(' Start to run the solver --  MIRL1 \n'); 
   fprintf(' --------------------------------------------\n');
   fprintf(' Iter       Error       CPUTime     Sparsity\n'); 
   fprintf(' --------------------------------------------\n');
end

for iter=1:itmax
        
    x0 = x; 
    w0 = w; 
    opts_ya.pho     = mu; 
    opts_ya.weights = w; 
    opts_ya.x0      = x0; 
    
    % call yall1 solver to solve the weighted L1-minimization
    x     = yall1(Aya, b, opts_ya);                  
    dx    = x-x0; 
    Error = sqrt(sum(dx.*dx))/max(sqrt(sum(x0.*x0)),1); 
    
    if disp  
       fprintf('%4d      %6.3e    %6.3fsec     %4d\n',...
                iter, Error, toc(t_start), nnz(abs(x)>=1e-3));        
    end

    if Error<1e-2 || iter==itmax                                % refinement
        T        = find(abs(x)>=1e-3);          
        if ~isa(A,'function_handle')       
            B    = A(:,T);
            x    = zeros(n,1);            
            x(T) = linsolve(B,b);
            Out.obj  = norm(A*x-b)^2/2;
        else
            x(abs(x)<1e-3)=0;
            Out.obj  = norm(A(x)-b)^2/2;
        end
        
        Out.sol  = x;
        Out.sp   = nnz(x);
        Out.iter = iter;         
        Out.time = toc(t_start);
        if  disp
        fprintf(' --------------------------------------------\n');
        end
        return;
    end    
    
    sx    = sort(abs(x),'descend');  
    eps2  = max(1e-3,sx(i0));  
    if isfield(pars,'s'); s = pars.s;                % update the sparsity
    else;                 s = sparsity(sx,rate);                              
    end
     
    theta = 1.005*theta;                  
    w     = ModWeight(x,abs(dx),theta,s,eps2);        % update the weight    
    beta  = sum(w0.*abs(x))/sum(w.*abs(x));
     
    if beta>1; mu = 0.2*mu;                     % update penalty parameter
    else ;     mu = beta*mu; 
    end        

end

end


%------------------Set Parameters----------------------------------------
function [itmax,rate,tol,disp,mu,i0,theta] = Get_parameters(m,n,At,b,opts)

if n<1000;      itmax=1000; else;  itmax = 100;          end
if log(n/m)<=1; rate=.7;    else;  rate  = 1/(log(n/m)); end 
if isfield(opts,'rate');    rate   = opts.rate;          end
if isfield(opts,'tol');     tol    = opts.tol;  else;  tol = 1e-4; end
if isfield(opts,'disp');    disp   = opts.disp; else;  disp = 1;    end 
if isa(At,'function_handle')
   mu = 0.01*max(abs(At(b)));
else    
   mu = 0.01*max(abs(At*b));  
end
i0    = ceil(m/(4*log(n/m))); 
theta = mu*m/n/10;

end
 
 
%------------------Modifeid weights----------------------------------------
function w = ModWeight(x,h,theta,k,eps2)
    n         = length(x);
    w         = ones(n,1);
    eps1      = 1e-10; 
    [~,Ind] = sort(h,'descend');       
    if k==0
        w=1./(abs(x)+eps2);
    else
        w(Ind(1:k))  = eps1+theta*sum(h(Ind(2:k+1)))/sum(h(Ind(1:k)));
        w(Ind(k+1:n))= eps1+theta+1./(abs(x(Ind(k+1:n)))+eps2);
   end  
end

%------------------Update the Sparsity-------------------------------------
function sp = sparsity(x,rate)
    rs=rate*sum(x); y=0; sp=0;   
    while y < rs
    sp=sp+1; y=y+x(sp,1); 
    end    
end

