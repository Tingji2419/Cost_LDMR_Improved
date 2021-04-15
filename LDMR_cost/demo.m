
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
total_cost=0;
ID=[];
for i = 1:n
%    if mod(i,10)==0
%        fprintf('%d / %d \n',i,n);
%    end
    y = ttdat(:,i);
    Xs = Proj*y;
    Es = y-trdat*Xs;
 %   c = cost(ttls,Xs, trdat, i, trls,imgsize);
 %   Es = Es .* c;
    
    [w] = Weight(y, trdat, Xs, trls, delta);
  %  [X] = LDMR(y, trdat, w, trls, Xs, Es, alpha, beta, imgsize);
    [X] = LDMR(y, trdat, w, trls, Xs, Es, alpha, beta, imgsize,i,ttls);
    [label] = classifier(trdat, X, trls, imgsize);  
    
    c1 = cost(X, trdat, i, trls,imgsize,ttls);
    total_cost = total_cost + c1;
    
    ID= [ID label];
    if mod(i,10)==0
        fprintf('%d / %d \n',i,n);
        
    end
end

acc = mean(ttls(:)==ID(:));
fprintf('Acc: %.2f\n',acc*100);

fprintf('total_cost: %d\n',total_cost);




