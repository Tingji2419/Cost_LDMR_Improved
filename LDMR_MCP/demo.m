
clear;clc
%
classnum = 38;
imgsize = [48 42];
%
% load training data
load('./data/YB_subset1.mat')
trdat = [];
trls = train_label;
for i=1:size(train_data, 2)
    t = reshape(train_data(:,i), [192 168]);
    t = imresize(t, imgsize);
    t = t(:);
    trdat = [trdat t];
end
clear train_data train_label

load('data/YB_subset4.mat')
ttdat = [];
ttls = train_label;
for i=1:size(train_data, 2)
    t = reshape(train_data(:,i), [192 168]);
    t = imresize(t, imgsize);
    t = t(:);
    ttdat = [ttdat t];
end
clear train_data train_label

n = size(ttdat, 2);
Proj = pinv(trdat'*trdat+0.01*size(trdat,2))*trdat';

alpha = 0.5;
beta  = 1e-3;
delta = 2;
lambda = 0.1;
%ID=[];


for gamma = 25:5:50
    ID=[];
for i = 1:n
%    if mod(i,10)==0
%        fprintf('%d / %d \n',i,n);
%    end
    y = ttdat(:,i);
    Xs = Proj*y;
    Es = y-trdat*Xs;
%    Es = Es .* cost(ttls,Xs, trdat, i, trls,imgsize);
    [w] = Weight(y, trdat, Xs, trls, delta);
%   [X] = LDMR(y, trdat, w, trls, Xs, Es, alpha, beta, imgsize);
    [X] = LDMR_MCP(y, trdat, w, trls, Xs, Es, alpha, beta, imgsize, gamma);
    [label] = classifier(trdat, X, trls, imgsize);  
    ID= [ID label];
    if mod(i,100)==0
        fprintf('%d / %d \n',i,n);
        
    end
end


acc = mean(ttls(:)==ID(:));
fprintf('gamma: %.2f\n',gamma);
fprintf('Acc: %.2f\n',acc*100);

end





