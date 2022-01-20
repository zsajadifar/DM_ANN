close all
clear all

load("breast.mat");

k =10;

%Kfold cross validation 
indices_K = crossvalind('Kfold',length(trainset),k);

neuronlist = [20,30,50];
methods = ["traingd","trainlm","trainbr"];
trainset = trainset';
labels_train = labels_train';
labels_train(labels_train==-1)=0;


accuracy_mat = zeros(length(methods),length(neuronlist));
sensitiviy_mat = zeros(length(methods),length(neuronlist));
FPR_mat = zeros(length(methods),length(neuronlist));

 m=1;
 for neurons= neuronlist
    j=1; 
    for method = methods
        accuracy= zeros(1,k);
        sensitivity= zeros(1,k);
        FPR= zeros(1,k);
        % K fold cross validation 
        for i = 1:k
            val_idx = (indices_K == i); 
            k_idx = ~val_idx;
            
            ptr = (trainset(:,k_idx)); 
            ttr = (labels_train(:,k_idx));
            pval = (trainset(:,val_idx )); 
            tval = (labels_train(:,val_idx ));

            net1 = feedforwardnet(neurons,method);

            net1.trainParam.epochs = 100;
            net1=train(net1,ptr,ttr); 

            tt = sim(net1, pval);
            tt (tt >mean(tt))=1;
            tt (tt <=mean(tt))=0;

            [acc,sens,spec] = performance(tt,tval);
            fprintf('The acc of method and neurons %d is %f \n',neurons, acc); 
            accuracy(1, i) = acc;
            sensitivity(1, i) = sens;
            FPR(1, i) = 1-spec;
        end

    accuracy_mat(j,m) = sum(accuracy)/k;
    sensitiviy_mat(j,m) = sum(sensitivity)/k;
    FPR_mat(j,m) = sum(FPR)/k;
    j = j + 1;

    end
    m = m + 1;   
 end