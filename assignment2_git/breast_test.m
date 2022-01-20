close all
clear all

load("breast.mat");

neuronlist = [30];
methods = ["trainlm"];
trainset = trainset';
labels_train = labels_train';
labels_train(labels_train==-1)=0;
labels_test(labels_test==-1)=0;
testset = testset';
labels_test = labels_test';

ptr = (trainset); 
ttr = (labels_train);

net = feedforwardnet(neuronlist,methods);

net.trainParam.epochs = 100;
net.divideFcn = 'dividerand';
net.divideParam.trainRatio=0.9;
net.divideParam.valRatio=0.1;
net.divideParam.testRatio=0;
%  net.trainParam.goal = 0.001; 
net=train(net,ptr,ttr); 

tt = sim(net, testset);
tt (tt >mean(tt))=1;
tt (tt <=mean(tt))=0;

[acc,sens,spec,TN,FP,FN,TP] = performance(tt,labels_test);
