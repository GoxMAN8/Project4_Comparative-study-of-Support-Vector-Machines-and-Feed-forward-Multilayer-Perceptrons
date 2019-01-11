%%% 1. Data Preperation %%%
[data, datatext, dataraw] = xlsread('C:\Users\Gedas\Desktop\City Uni\INM427. Neural computing\PROJECT\MATLAB\NNdatasetv2_SVM.xlsx');

data20k=data(1:20000, :); %
trainData20k=StatisticalNormaliz(data20k(1:20000, 3:end),'standard');
trainTarg20k=data20k(1:20000, 2);
[trainInd,valInd, testInd] = dividerand(length(data20k),0.8,0.2,0.0); % Creates indices for 2 way split. 80%\20%

trainData = trainData20k(trainInd, :); % Splits the data according to the generated index
trainTarg=trainTarg20k(trainInd, :);
valData=trainData20k(valInd, :);
valTarg=trainTarg20k(valInd, :);

%%% SET PARAMETERS for tone layer feedfoward network %%%
%GRIDS%
gridRBF = 2.^(-10:1:5); % Exponential growth as suggested by SVM guide paper (2003)
gridC = 2.^(-5:1:10); %Exponential growth as suggested by SVM guide paper (2003)
CV = 10; % 10 fold cross validation. 
%%% PARAMETERS SET %%%

%%% 2. Cross validated/grid searched model (validation data) %%%

% Grid search for best RBF_Sigma starts
bestSVM = struct('SVMModel', NaN, 'C', NaN, 'KernelScale', NaN, 'Error', Inf); %Stores Best SVM model

% Grid search for best RBF_Sigma/Kernel Scale 
count=0;
t1=datetime('now');
for RBF_Sigma = gridRBF
% Grid search for best C/Box Constraing
    bestCSVM = struct('SVMModel', NaN, 'C', NaN, 'KernelScale', NaN, 'Error', Inf); %Stores Best C model
   
    for C = gridC
        t3=datetime('now');
        fprintf('TESTING PARAMETERS: C:%d & RBF_Sigma:%d \n', C, RBF_Sigma)
        anSVMModel = fitcsvm(valData, valTarg, 'KernelFunction', 'RBF', ...
            'BoxConstraint', C, 'KernelScale', RBF_Sigma, 'Standardize', false, 'CrossVal', 'on', 'KFold', CV); % MODEL
        L=kfoldLoss(anSVMModel); % Stores classification error values
        fprintf('CROSS-VALIDATED ACCURACY: %d\n', 1-L)
        t4=datetime('now');
        fprintf('...TIME COST: %s\n', t4-t3)
        count=count+1;
        fprintf(' %d Combinations tested...\n', count)
        fprintf('----------------------------------------------\n')
        % Saving best SVM for C selection
        if L < bestCSVM.Error
            bestCSVM.Error = L;
            bestCSVM.C=C;
            bestCSVM.SVMModel=anSVMModel;
        end
    end
    %Saving best SVM for RBF_Sigma selection
    if (bestCSVM.Error < bestSVM.Error)
        bestSVM.Error=bestCSVM.Error;
        bestSVM.SVMModel = bestCSVM.SVMModel;
        bestSVM.KernelScale = RBF_Sigma;
        bestSVM.C =bestCSVM.C;
    end
end
t2=datetime('now');
durGridSearch=t2-t1; % Check the duration of a gridsearch

%%% 3. Learning curves (training data) %%%

t5=datetime('now');
[train_size,train_error,test_error] = learningCurve(trainData, trainTarg, 1000, bestSVM.C, bestSVM.KernelScale);
t6=datetime('now');
durLearnCurve=t6-t5;
train_accuracy=1-train_error;
test_accuracy=1-test_error;

figure1=figure;
plot(train_size,train_accuracy,train_size,test_accuracy)
title('Learning curves')
xlabel('Sample Size')
ylabel('Classification accuracy')
legend('Training data','Cross-validated data')
saveas(figure1,'learningcurvesSVM.jpg')

%%% 4. Fit SVM model , once more on training data and best parameters %%%
SVMmodel=fitcsvm(trainData, trainTarg, 'KernelFunction', 'RBF', ...
                'BoxConstraint', bestSVM.C, 'KernelScale', bestSVM.KernelScale, 'Standardize', false, 'CrossVal', 'on', 'KFold', CV );

[label, score] = kfoldPredict(SVMmodel); % Make predictions

%%% 5. Confusion matrix and f1 scores. %%%
C = confusionmat(trainTarg,label);

% Convert to two class data columns
ytrue=zeros(length(trainTarg),1);
ypred=zeros(length(label),1);

for x = 1  : length(trainTarg)
   if testTarg(x)==1
     ytrue(x) = 1;
   else
     ytrue(x) = 0;
   end
end

for x = 1  : length(label)
   if label(x)==1
     ypred(x) = 1;
   else
     ypred(x) = 0;
   end
end
% End of conversion.

% Confusion matrix and f1 scores (continued)
plotconfusion(transpose(ytrue), transpose(ypred));

%%% 6. Precision/Recall/F1 %%%
precision = @(C) diag(C)./sum(C,2);
recall = @(C) diag(C)./sum(C,1)';
f1Scores = @(C) 2*(precision(C).*recall(C))./(precision(C)+recall(C));
meanF1 = @(C) mean(f1Scores(C));


%%% 7. AUC-ROC curves %%%
%SVMModel = fitPosterior(SVMmodel);
%[~,scores1] = resubPredict(SVMModel);
%[x1,y1,~,auc1] = perfcurve(trainTarg,scores1(:,2),1);

%plot(x1,y1)
%xlabel('False positive rate'); ylabel('True positive rate');
%title('ROC for classification by SVM');
%fprintf('AUC: %d\n', auc1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% BONUS. Fit on the whole data %%%
[data, datatext, dataraw] = xlsread('C:\Users\Gedas\Desktop\City Uni\INM427. Neural computing\PROJECT\MATLAB\NNdatasetv2_SVM.xlsx');
trainDataFULL=StatisticalNormaliz(data(:, 3:end),'standard');
trainTargFULL=data(:, 2);

t7=datetime('now');
CV=5;
SVMmodelFULL=fitcsvm(traiDataFULL, trainTargFULL, 'KernelFunction', 'RBF', ...
                'BoxConstraint', bestSVM.C, 'KernelScale', bestSVM.KernelScale, 'Standardize', false, 'CrossVal', 'on', 'KFold', CV );
t8=datetime('now');
durFullTrain=t8-t7;

[labelF, scoreF] = kfoldPredict(SVMmodelFULL); % Make predictions

CF = confusionmat(trainTargFULL,labelF); % Confusion matrix

precisionF = @(CF) diag(CF)./sum(CF,2);
recallF = @(CF) diag(CF)./sum(CF,1)';
f1ScoresF = @(CF) 2*(precision(CF).*recall(CF))./(precision(CF)+recall(CF));
meanF1F = @(CF) mean(f1Scores(CF));

fprintf('---DONE---')


