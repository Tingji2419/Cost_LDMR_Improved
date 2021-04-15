function [c] = cost(Xs, trdat, i,trls,imgsize,ttls)
%2021_2
c = 1;
Cig = 20;
Cgi = 2;
Cgg = 1;
Cii = 1;
    
[label, ~] = classifier(trdat, Xs, trls, imgsize);

if label < 9 && ttls(i) > 8
    c = Cig; 
end
if label < 9 && ttls(i) < 9
    c = Cgg;
end
if label > 8 && ttls(i) < 9
    c = Cgi;
end
if label > 8 && ttls(i) > 8
    c = Cii;
end

end