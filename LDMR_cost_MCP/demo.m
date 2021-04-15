
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

gamma = 3:2:21;
boundary = round(1/11 * classnum);
total_cost = 0;
ID=zeros(1,n);
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
fprintf('boundary: %.2f\n',boundary);
fprintf('cost: %.2f\n',total_cost / n);








