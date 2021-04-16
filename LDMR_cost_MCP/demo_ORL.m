clear;clc
%
classnum = 40;
imgsize = [32 32];
%
% load training data
load('./data_ORL/ORL_32x32.mat')
trdat = [];
trls = [];
for i=1:classnum
    x = randperm(10);
    x = x(1:8);
    t = reshape(fea(x.*i, :), imgsize);
    t = t(:);
    trdat = [trdat t];
    trls = [trls gnd(x)];
end
% clear train_data train_label
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


boundary = round(1/10 * classnum);

for gamma = 1:1:20
total_cost = 0;
ID=zeros(1,n);
tic
for i = 1:n
    y = ttdat(:,i);
    Xs = Proj*y;
    Es = y-trdat*Xs;
    [w] = Weight(y, trdat, Xs, trls, delta);
   % [X] = LDMR_MCP_cost(y, trdat, w, trls, Xs, Es, alpha, beta, imgsize, gamma, i, ttls, boundary);
    [X] = LDMR(y, trdat, w, trls, Xs, Es, alpha, beta, imgsize);
    [label] = classifier(trdat, X, trls, imgsize);  
    ID(i)=label;
    if mod(i,100)==0
        fprintf('%d / %d \n',i,n);    
    end
    c1 = cost(X, trdat, i, trls,imgsize,ttls, boundary);
    total_cost = total_cost + c1;
end
acc = mean(ttls(:)==ID(:));
fprintf('Acc: %.2f\n',acc*100);
fprintf('gamma: %.2f\n',gamma);
fprintf('cost: %.2f\n',total_cost / n);
toc
end








