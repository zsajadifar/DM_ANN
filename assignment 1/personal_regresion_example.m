clc
clear 
close all

load('data_personal_regression_problem.mat');

%% dataset
%r0827611
d1=8;d2=7;d3=6;d4=2;d5=1;
Tnew = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)./(d1+d2+d3+d4+d5);
[~ ,indx] = datasample(1:13600,3000,'Replace',false);

X = [X1(indx),X2(indx)];
Y = Tnew(indx);

% X = X(1:2000,:);
% Y = Y(1:2000,:);

%% visualize training dataset
X1_train   = X1(indx(1:1000));
X2_train   = X2(indx(1:1000));
X_train    = [X1_train X2_train];
Tnew_train = Tnew(indx(1:1000));

X1_valid   = X1(indx(1001:2000));
X2_valid   = X2(indx(1001:2000));
X_valid    = [X1_valid X2_valid];
Tnew_valid = Tnew(indx(1001:2000));

X1_test   = X1(indx(2001:3000));
X2_test   = X2(indx(2001:3000));
X_test    = [X1_test X2_test];
Tnew_test = Tnew(indx(2001:3000));

x = linspace (min(X1_train),max(X1_train),1000);
y = linspace (min(X2_train),max(X2_train),1000);
[XX,YY] = meshgrid(x,y);
z = griddata(X1_train,X2_train,Tnew_train,XX,YY,'cubic');
mesh(x,y,z)
hold on,
plot3(X1_train,X2_train,Tnew_train,'.')
title('Train dataset')
xlabel('X1')
ylabel('X2')
zlabel('Target')
grid on

%% build and train neural network, 2layer

%algorithm = 'trainlm';
algorithm = 'trainbr';
epoch = 1000; 
% neurons = [10,20,30,40,50];
neurons =20;
iter = 5;
N = numel(Tnew_test);
time_alg1=[];
RMSE=[];

for j=1:numel(neurons)
    for i=1:iter
        net=[];
        n_neurons_l1 = neurons(j);
        n_neurons_l2 = neurons(j);
        structure = [n_neurons_l1, n_neurons_l1] ;  
        structure(structure == 0) = [] ; 

        net=feedforwardnet(structure, algorithm) ;
        net = configure(net,X',Y');
        net.trainParam.epochs = epoch;
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = (1:1000);
        net.divideParam.valInd=(1001:2000);
        net.divideParam.testInd=(2001:3000);

        tic ;                                       
        [net,tr] = train(net,X',Y') ;                   
        time_alg1(j,i) = toc ;
        
        res_test =sim(net,X_valid');
        
        RMSE(j,i)= sqrt(sum((res_test-Tnew_valid').^2)/N);
    end
end
% disp(['Training time for ' algorithm ': ' ...
%     num2str(time_alg1) 's']) ;  

total_Time = mean(time_alg1,2);
total_rmse = mean(RMSE,2);
% figure,plot(neurons,total_rmse)
% title('RMSE vs Number of Hidden Units')
% xlabel('neurons')
% ylabel('RMSE')

%% performance on testset and visualization

figure,
subplot(1,2,1)
x = linspace (min(X1_test),max(X1_test),1000);
y = linspace (min(X2_test),max(X2_test),1000);
[XX,YY] = meshgrid(x,y);
z = griddata(X1_test,X2_test,Tnew_test,XX,YY,'cubic');
mesh(x,y,z)
hold on,
plot3(X1_test,X2_test,Tnew_test,'.')
title('Test dataset')
xlabel('X1')
ylabel('X2')
zlabel('Target')


subplot(1,2,2)
z = griddata(X1_test,X2_test,res_test',XX,YY,'cubic');
mesh(x,y,z)
title('Approximated Test dataset')
xlabel('X1')
ylabel('X2')
zlabel('Approximated Target')






