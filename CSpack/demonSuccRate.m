% This code demonstrates the success recovery rate in compressed sensing
clc; clear; close all; 

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
    rate    = 0; 
    switch  test
    case 1; s   = sm(j);
    case 2; m   = ceil(sm(j)*n);
    end    
    for S       = 1:noS         
        A       = randn(m,n);
        I0      = randperm(n); 
        I       = I0(1:s);
        xopt    = zeros(n,1);
        xopt(I) = randn(s,1); 
        data.A  = normalization(A, 3); 
        data.At = data.A';                
        data.b  = data.A(:,I)*xopt(I);   
        
        pars.s  = s;
        out     = CSsolver(data,n,'NHTP',pars); clc; SucRate     
        rate    = rate + (norm(out.sol-xopt)/norm(xopt)<1e-2); 
    end
    clc; SucRate  = [SucRate rate]  
    
    figure(1)
    set(gcf, 'Position', [1000, 200, 400 350]);
    xlab = {'s','m/n'};
    plot(sm(1:j),SucRate/noS,'r*-','LineWidth',1), 
    xlabel(xlab{test}), ylabel('Success Rate') 
    axis([min(sm) max(sm) 0 1]); grid on; 
    legend('NHTP','Location','NorthEast'); hold on, pause(0.1)
    
end


