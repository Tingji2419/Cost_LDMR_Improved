function [Xn] = LDMR_MCP_cost(y, A, D, trls, Xinit, Einit, alpha, beta, imgsize, gamma, No_p, ttls, boundary)

mu = 1;
eps_abs = 1e-3;
eps_rel = 1e-3;
Max_Iter = 50;
p = imgsize(1);
q = imgsize(2);
n = size(A,2);
Xn = Xinit;
En = Einit;
Zn = zeros(prod(imgsize),1);

classnum = length(D);
WA = [];
for i=1:length(trls)
    lb = trls(i);
    WA = [WA D(lb)*A(:,i)];
end
M1 = zeros(n, n);
M2 = zeros(n, n);
nc = 0;
for i=1:classnum
   pos = find(trls==i);
   M1(pos,pos) = D(i)*A(:,pos)'*A(:,pos);
   M2(pos,pos) = (D(i)*A(:,pos))'*(D(i)*A(:,pos));
   nc  = nc + length(pos);
end
M = pinv(alpha/mu*M1+beta/mu*(WA'*WA)+A'*A-beta/mu*M2)*A';

for iter = 1:Max_Iter  
    Zo = Zn;
    Xo = Xn;
    Eo = En;
   
    % update E
     m1 = reshape( cost(Xo, A, No_p, trls, imgsize, ttls, boundary).* (A*Xo - y + Zo/mu), imgsize);  
     
     Em  =  proximal_matrix(m1, 1/mu, gamma);
     En   =  reshape(Em, [prod(imgsize), 1]);
     %
     
    % update X
    g = y + En - Zo/mu;
    Xn = M*g;       
    
    % update Z
    Zn = Zo + mu*(A*Xn - En - y);
    
    % check the convergence condition
    eps_pri = sqrt(p*q)*eps_abs + eps_rel*max( max(norm(A*Xn,2),norm(En,2)), norm(y,2));
    eps_dual = sqrt(n)*eps_abs+eps_rel*norm(A'*Zn, 2);
    r = A*Xn - y - En;
    s = mu*A'*(En-Eo);
    convergence = (norm(r,2)<eps_pri) && (norm(s,2)<eps_dual);
    if convergence
        break;
    end  
    
end

end

function out = proximal_matrix(Y, lambda, gamma)
[U, S, V] = svd(Y, 'econ');
s = diag(S);
n = length(s);
for i=1:n
    u = s(i);
    if ( u<gamma && u>=lambda )
        u = (u-lambda)/(1-lambda/gamma);
    elseif (u<lambda)
        u = 0;
    else
        u = s(i);
    end
    s(i) = u;   
end
out = U*diag(s)*V';

end


