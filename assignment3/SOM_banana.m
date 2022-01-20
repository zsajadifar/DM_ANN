close all;
clear all;
clc;


load banana.mat
X = X';
% X(3,:)=Y;

net = newsom(X,[4 4],'hextop','linkdist'); 
% hextop , gridtop , randtop    linkdist, dist , mandist 

% plot the data distribution with the prototypes of the untrained network
figure;
subplot(1,2,1),plot3(X(1,:),X(2,:),Y,'.g','markersize',10);
axis([-0.5 1.5 -0.5 1.5]);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off


% finally we train the network and see how their position changes
net.trainParam.epochs = 400;
net = train(net,X);
subplot(1,2,2),plot3(X(1,:),X(2,:),Y,'.g','markersize',10);
axis([0 1 0 1]);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off



 