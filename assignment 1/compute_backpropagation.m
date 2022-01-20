W=[-0.2, 1.3]  ;
beta = -0.5;
V =[-0.2 , -0.7 ,0.5 ; -0.8 ,0.6 , 0.4];  
X1=[1 1 1]'; 
X2=[2 -1 1]';  
X3=[3 0 1]'; 
X=[X1,X2,X3];
Y = [1,-1,-1];

y_prime=W*(sigma(V*X)) + beta  ;


dw1 = sum((y_prime-Y).*sigma(V(1,:)*X));
dw2 = sum((y_prime-Y).*sigma(V(2,:)*X));
dbeta = sum((y_prime-Y));

dv1 = sum((y_prime-Y).*W(1,1).*sigma(V(1,:)*X).*(1-sigma(V(1,:)*X)).*X(1,:));
dv2 = sum((y_prime-Y).*W(1,1).*sigma(V(1,:)*X).*(1-sigma(V(1,:)*X)).*X(2,:));
dv3 = sum((y_prime-Y).*W(1,1).*sigma(V(1,:)*X).*(1-sigma(V(1,:)*X)));

dv4 = sum((y_prime-Y).*W(1,2).*sigma(V(2,:)*X).*(1-sigma(V(2,:)*X)).*X(1,:));
dv5 = sum((y_prime-Y).*W(1,2).*sigma(V(2,:)*X).*(1-sigma(V(2,:)*X)).*X(2,:));
dv6 = sum((y_prime-Y).*W(1,2).*sigma(V(2,:)*X).*(1-sigma(V(2,:)*X)));

dv = [dv1,dv2,dv3;dv4,dv5,dv6];
dw = [dw1 , dw2];

w_new = W - dw;
v_new = V - dv;
beta_new = beta - dbeta;




