function [sci, csfL, csfM, csfH] = getSCINxN( img, N,P)

D = zeros(N,N,N,N);

%create bases
basis = dctmtx(N);
basisT = basis';
for y = 1:N
    for x = y:N
        D(:,:,y,x) = (basisT(:,y)*basis(x,:));
    end
end

D =D*N;

[nH nW] = size(img);

Dext = zeros(nH,nW,N,N);
for y= 1:N
    for x = 1:N
        if( (y == 1 && x == 1))
        elseif( (y <= x))
            Dext(:,:,y,x) = conv2(img, fliplr(D(:,:,y,x)),'same');            
        else
            Dext(:,:,y,x) = conv2(img, fliplr(D(:,:,x,y)'),'same');            
        end
        
        Dext(:,:,y,x) =  P + abs(Dext(:,:,y,x));
       
   end
end

m2 = zeros(nH,nW);
m0 = zeros(nH,nW);
m4 = zeros(nH,nW);

for y= 1:N
    for x = 1:N
        dist = ((y-1)^2+(x-1)^2);
        m0 = m0 + Dext(:,:,y,x);
        m2 = m2 + dist.*Dext(:,:,y,x);
        m4 = m4 + (dist^2).*Dext(:,:,y,x);
    end
end

%inverse sci 
sci = (m4+ eps)./(m2.*m2+eps);
weight = 1./(m0 + eps);
csfL = (Dext(:,:,1,2) + Dext(:,:,2,1)).*weight;
csfM = (Dext(:,:,2,2) + Dext(:,:,1,3) + Dext(:,:,3,1)+ Dext(:,:,2,3) + Dext(:,:,3,2)).*weight;
csfH = (Dext(:,:,4,1) + Dext(:,:,1,4) + Dext(:,:,4,2)+ Dext(:,:,2,4) + Dext(:,:,3,3) + Dext(:,:,4,3) + Dext(:,:,3,4)+ Dext(:,:,4,4)).*weight;


return;