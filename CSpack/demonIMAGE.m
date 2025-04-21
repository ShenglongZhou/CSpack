% demon sparse compressed sensing problems with image data
clc; clear; close all; addpath(genpath(pwd));

Img       = phantom(64*4); % 64*4 or 64*8
sigma     = 0.01;          % 0.05 or 0.01       
nsam      = 60;            % 40 or 60 

% generate data (A, At, b), all of them are function handle 
[A,At,b,be,~,~,out0] = getrealdata(Img,nsam,sigma,0);      
n                    = size(out0.I,2)^2;

% set parameters for 'NL0R'
pars.tau    = sigma;
pars.lambda = 0.015;   
pars.obj    = 2*norm(b-be)^2/4;
s           = [];
solver      = { 'NL0R', 'PSNP', 'NHTP', 'GPNP', 'IIHT'};
for alg     = 1:length(solver)
    out{alg}    = CSsolver(A,At,b,n,s,solver{alg},pars); 
    pars.lambda = 0.001;
    pars.q      = 0.5;
    s           = nnz(out{alg}.sol); 
end

% display results 
figure('Renderer', 'painters', 'Position', [900 200 600 400])
subplot(2,3,1), imagesc(out0.I),colormap(gray)
title('Original Image'), box off, axis off

for alg = 1:length(solver)
    x2d   = reshape(out{alg}.sol,size(out0.I));
    Ibar  = out0.Ibar;
    x2d   = out0.W(x2d) +  Ibar;
    subplot(2,3,alg+1), imagesc(x2d), colormap(gray) 
    psnrx(5) = psnr(out0.I,x2d, max(out0.I(:))- min(out0.I(:)));  
    ax = gca; axis(ax,'off');
    title(ax,[solver{alg},': PSNR = ',num2str(psnrx(5),'%2.2f')]);  
    ax.XLabel.Visible = 'on'; 
end 
