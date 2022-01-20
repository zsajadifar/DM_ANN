%% Reconstruct Handwritten Digit Images Using Sparse Autoencoder  

%% 
% Load the training data. 
XTrain = digitTrainCellArrayData;

%%
% The training data is a 1-by-5000 cell array, where each cell containing
% a 28-by-28 matrix representing a synthetic image of a handwritten digit.  

%% 
% Train an autoencoder with a hidden layer containing 50 neurons. 
hiddenSize = 300;
autoenc = trainAutoencoder(XTrain,hiddenSize,...
        'MaxEpochs', 250, ...
        'L2WeightRegularization',0.004,...
        'SparsityRegularization',2,...
        'SparsityProportion',0.9);  

%% 
% Load the test data. 
XTest = digitTestCellArrayData; 

%%
% The test data is a 1-by-5000 cell array, with each cell containing a 28-by-28
% matrix representing a synthetic image of a handwritten digit.  

%% 
% Reconstruct the test image data using the trained autoencoder, |autoenc|. 
xReconstructed = predict(autoenc,XTest);  

%% 
% View the actual test data. 
figure;
for i = 1:20
    subplot(4,5,i);
    % select samples with the corresponding indexes
    idx = 100+200*i;
    imshow(XTest{idx});
end     

%% 
% View the reconstructed test data. 
figure;
for i = 1:20
    subplot(4,5,i);
    % select samples with the corresponding indexes
    idx = 100+200*i;
    imshow(xReconstructed{idx});
end 

for i=1:5000
    error(i)= mean(mean((XTest{i}-xReconstructed{i}).^2));
%     mseError(i) = mse(XTest{i}-xReconstructed{i});
end
final_error=mean(error);
% final_mseError=mean( mseError);

%% 
% Copyright 2012 The MathWorks, Inc.