% This code demonstrates the success recovery rate in compressed sensing
clc; clear; close all; addpath(genpath(pwd));

test    = 1; % =1 (succrate v.s. s); =2 (succrate v.s. m/n)
n       = 256; 
m       = ceil(0.25*n);
s       = ceil(0.05*n);
noS     = 100;
switch  test
case 1; sm = 10:2:38;
case 2; sm = linspace(0.06,0.28,12);
end    
    
SucRate     = [];
pars.disp   = 0;
for j       = 1:length(sm)
    rate    = [0; 0]; 
    switch  test
    case 1; s   = sm(j);
    case 2; m   = ceil(sm(j)*n);
    end    
    for S       = 1:noS       
        I      = randperm(n,s); 
        xopt    = zeros(n,1);
        xopt(I) = randn(s,1); 
        A       = normalization(randn(m,n), 3);              
        b       = A(:,I)*xopt(I);   
        out     = CSsolver(A,[],b,n,s,'NHTP',pars); clc; SucRate   
        rate(1) = rate(1) + (norm(out.sol-xopt)/norm(xopt)<1e-2);
        out     = CSsolver(A,[],b,n,s,'GPNP',pars); clc; SucRate   
        rate(2) = rate(2) + (norm(out.sol-xopt)/norm(xopt)<1e-2); 
    end
    clc; SucRate  = [SucRate rate]  
    
    figure(1)
    set(gcf, 'Position', [1000, 200, 400 350]);
    xlab = {'s','m/n'};
    plot(sm(1:j),SucRate(1,:)/noS,'r*-'); hold on
    plot(sm(1:j),SucRate(2,:)/noS,'bo-'); hold on
    xlabel(xlab{test}), ylabel('Success Rate') 
    axis([min(sm) max(sm) 0 1]); grid on; 
    legend('NHTP','GPSP','Location','NorthEast'); 
    hold on, pause(0.1)    
end
