% demon sparse compressed sensing problems with image data
clc; clear all ;close all; warning off 

Img       = phantom(64*4); % 64*4 or 64*8
sigma     = 0.01;          % 0.05 or 0.01       
nsam      = 60;            % 40 or 60 

% generate the testing data (data.A, data.At, data.b), 
% all of them are function handle 
[data.A,data.At,data.b,be,~,~,out0] = getrealdata(Img,nsam,sigma,0);      
n         = size(out0.I,2)^2;

% set parameters for 'NL0R'
pars.tau  = sigma;
pars.lam  = 0.01;   
pars.obj  = 2*norm(data.b-be)^2/3;
out       = CSsolver(data,n,'NL0R',pars); 

% results output 
fprintf(' CPU time:       %.3fsec\n', out.time);
fprintf(' Sparsity:       %.2d\n', nnz(out.sol));
fprintf(' Objective:      %5.2e\n', out.obj);
fprintf(' Sample size:    %dx%d\n',length(data.b),n);

% display results 
figure('Renderer', 'painters', 'Position', [800, 200, 600 250])
subplot(1,2,1), imagesc(out0.I),colormap(gray)
title('Original Image'), box off, axis off

x2d   = reshape(out.sol,size(out0.I));
Ibar  = out0.Ibar;
x2d   = out0.W(x2d) +  Ibar;
subplot(1,2,2), imagesc(x2d), colormap(gray) 
psnrx(5) = psnr(out0.I,x2d, max(out0.I(:))- min(out0.I(:)));  
ax = gca; axis(ax,'off');
title(ax,['NL0R: PSNR = ',num2str(psnrx(5),'%2.2f')]);  
ax.XLabel.Visible = 'on'; 

 
 
