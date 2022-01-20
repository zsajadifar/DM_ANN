clc
clear all
close all

X=randn(50,500);
mu = mean(X,2);
X_norm = X-mu;
[eigvals,eigvec] = linearpca(X_norm');

for i=1:49
    E = eigvec(:, 1:i);
    z = E'*X_norm;
    X_hat = E * z + mu;
    RMSE(i) = sqrt(mean(mean((X-X_hat).^2)));
end

plot(RMSE,'LineWidth',2)
xlabel('q')
ylabel('RMSE')


%% Correlated data
clear
load choles_all
mu = mean(p,2);
sig = std(p,[],2);
p_norm = (p - mu) ./ sig;
[eigvals,eigvec] = linearpca(p_norm');

for i=1:20
    E = eigvec(:, 1:i);
    z = E'*p_norm;
    p_hat = (E * z).*sig + mu;
    RMSE(i) = sqrt(mean(mean((p-p_hat).^2)));
end

plot(RMSE,'LineWidth',2)
xlabel('q')
ylabel('RMSD')

%% 3.2.3

clear
x=randn(50,500);
x = mapstd(x);
maxfrac = 0.01;
[y,PS]=processpca(x,maxfrac);
[x_hat]=processpca('reverse',y,PS);
RMSE = sqrt(mean(mean((x-x_hat).^2)));


i=1;
for maxfrac=[0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001]
    [y,PS]=processpca(x,maxfrac);
    [x_hat]=processpca('reverse',y,PS);
    RMSE(i) = sqrt(mean(mean((x-x_hat).^2)));
    i=i+1;
end

plot(RMSE,'LineWidth',2)
xlabel('maxfrac')
ylabel('RMSD')


clear 
load choles_all
x = mapstd(p);
maxfrac = 0.1;
[y,PS]=processpca(x,maxfrac);
[x_hat]=processpca('reverse',y,PS);
RMSE = sqrt(mean(mean((x-x_hat).^2)));

i=1;
for maxfrac=[0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001]
    [y,PS]=processpca(x,maxfrac);
    [x_hat]=processpca('reverse',y,PS);
    RMSE(i) = sqrt(mean(mean((x-x_hat).^2)));
    i=i+1;
end
plot(RMSE,'LineWidth',2)
xlabel('maxfrac')
ylabel('RMSD')


