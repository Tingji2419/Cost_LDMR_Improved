function [c] = cost(Xs, trdat, i, trls, imgsize, ttls, boundary)
%2021_2
c = 1;
Cig = 20;
Cgi = 2;
Cgg = 1;
Cii = 0;
[label, ~] = classifier(trdat, Xs, trls, imgsize);

% imposter: > boundary-1

if label < boundary && ttls(i) > (boundary - 1)
    c = Cig;   
end
if label < boundary && ttls(i) < boundary
    c = Cgg;
end
if label > (boundary - 1) && ttls(i) < boundary
    c = Cgi;
end
if label > (boundary - 1) && ttls(i) > (boundary - 1)
    c = Cii;
end

end