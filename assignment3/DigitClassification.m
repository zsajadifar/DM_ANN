clear all
close all
nntraintool('close');
nnet.guis.closeAllViews();

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
% rng('default')


% Load the training data into memory
%[xTrainImages, tTrain] = digittrain_dataset;
load('digittrain_dataset');

% Layer 1
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);

figure;
plotWeights(autoenc1);
feat1 = encode(autoenc1,xTrainImages);

% Layer 2
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);

% % Layer 3
% hiddenSize3 = 25;
% autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
%     'MaxEpochs',50, ...
%     'L2WeightRegularization',0.002, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.1, ...
%     'ScaleData', false);
% 
% feat3 = encode(autoenc3,feat2);

% Layer 4
% softnet = trainSoftmaxLayer(feat3,tTrain,'MaxEpochs',400);
softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);


% Deep Net
% deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);
deepnet = stack(autoenc1,autoenc2,softnet);


% Test deep net
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;
%[xTestImages, tTest] = digittest_dataset;
load('digittest_dataset');
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
y = deepnet(xTest);
figure;
plotconfusion(tTest,y);
classAcc=100*(1-confusion(tTest,y))


% Test fine-tuned deep net
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end
deepnet = train(deepnet,xTrain,tTrain);
y = deepnet(xTest);
figure;
plotconfusion(tTest,y);
classAcc=100*(1-confusion(tTest,y))
view(deepnet)

%Compare with normal neural network (1 hidden layers)
for i=1:20
    for j=50:20:130
        net = patternnet([j]);
        net=train(net,xTrain,tTrain);
        y=net(xTest);
        plotconfusion(tTest,y);
        classAcc=100*(1-confusion(tTest,y));
    %     view(net)
        result(j,i)=classAcc;
    end
end
mu = mean(result,2);

% %Compare with normal neural network (2 hidden layers)
%