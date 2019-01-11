%%%% 1. Data Preperation %%%
[data, datatext, dataraw] = xlsread('C:\Users\Gedas\Desktop\City Uni\INM427. Neural computing\PROJECT\MATLAB\NNdatasetv2_SVM.xlsx');

data20k=data(1:20000, :); %
trainData20k=StatisticalNormaliz(data20k(1:20000, 3:end),'standard');
trainTarg20k=data20k(1:20000, 2);
[trainInd,valInd, testInd] = dividerand(length(data20k),0.8,0.2,0.0); % Creates indices for 2 way split. 80%\20%

trainData = trainData20k(trainInd, :); % Splits the data according to the generated index
trainTarg= trainTarg20k(trainInd, :);
valData= trainData20k(valInd, :);
valTarg= trainTarg20k(valInd, :);


%%% 2. Cross-validated/grid search (validation data) %%%
% ONE HIDDEN LAYER FEED-FORWARD NETWORK %

%%% SET PARAMETERS for tone layer feedfoward network %%%
%GRIDS%
gridH=250; % Hiddden unit grid.( h )(SET IT) 
gridE=750; % Number of epochs grid.(numUpdates) (SET IT)
gridL=0.99; % Learning rate decay. (alpha) (SET IT)
gridM=0.1; % Momentum factor grid. (mu) (SET IT)
gridA=[0.6, 0.8]; % Learning rate grid. (etaInit (SET IT)
CV = 10; % 10 fold cross validation. 
%%% PARAMETERS SET %%%

    
%Grid search for optimal number of neurons in hidden layer
bestNN1L = struct('h', NaN, 'numUpdates', NaN, 'alpha', NaN, 'mu', NaN, 'etaInit', NaN,...
    'Error', Inf); %Stores Best NN model    

t1=datetime('now');
count=0;
for H=gridH
    %Grid search for optimal number of epochs
    bestECombo=struct('h', NaN, 'numUpdates', NaN, 'alpha', NaN, 'mu', NaN, 'etaInit', NaN,...
        'Error', Inf);

    for E=gridE
        % Grid search for best regularization cost (second loop)
        bestLCombo = struct('h', NaN, 'numUpdates', NaN, 'alpha', NaN, 'mu', NaN, 'etaInit', NaN,...
            'Error', Inf);

        for L = gridL
            % Grid search for best M/Momentum factor (inner loop)
            bestMCombo = struct('h', NaN, 'numUpdates', NaN, 'alpha', NaN, 'mu', NaN, 'etaInit', NaN,...
                'Error', Inf);

            for M = gridM
                % Grid search for best L/Learning rate
                bestACombo = struct('h', NaN, 'numUpdates', NaN, 'alpha', NaN, 'mu', NaN, 'etaInit', NaN,...
                    'Error', Inf);

                for A=gridA
                    conf.h=H; % Hidden units
                    conf.numUpdates=E; % Max epochs
                    conf.alpha=L; % Learning decay
                    conf.mu=M; % Momentum factor
                    conf.etaInit=A; % Learning rate
                    t3=datetime('now');
                    fprintf('TESTING PARAMETERS: H:%d & E:%d & L:%d & M:%d & A:%d \n', H,E,L,M,A)
                    [errorCV, accuracyCV,~,~,~,~]=nnTrain1L(conf, CV, valData, valTarg); % MODEL
                    fprintf('CROSS-VALIDATED ACCURACY: %d\n', accuracyCV)
                    t4=datetime('now');
                    fprintf('...TIME COST: %s\n', t4-t3)
                    count=count+1;
                    fprintf(' %d Combinations tested...\n', count)
                    fprintf('----------------------------------------------\n')
                % Saving best NN for A selection
                    if errorCV < bestACombo.Error
                        bestACombo.Error = errorCV;
                        bestACombo.etaInit=A;
                    end
                end

                %Saving best NN for M selection
                if (bestACombo.Error < bestMCombo.Error)
                    bestMCombo.Error=bestACombo.Error;
                    bestMCombo.etaInit = bestACombo.etaInit;
                    bestMCombo.mu = M;
                end
            end

        % Saving the best NN for L selection
           if bestMCombo.Error < bestLCombo.Error
              bestLCombo.etaInit =  bestMCombo.etaInit;
              bestLCombo.mu = bestMCombo.mu;
              bestLCombo.alpha= L;
              bestLCombo.Error = bestMCombo.Error;
           end
        end

        %Saving the best NN for E selection
        if bestLCombo.Error < bestECombo.Error
            bestECombo.etaInit=bestLCombo.etaInit;
            bestECombo.mu=bestLCombo.mu;
            bestECombo.alpha=bestLCombo.alpha;
            bestECombo.numUpdates=E;
            bestECombo.Error=bestLCombo.Error;
        end
    end

    %Saving the best NN for overall model
    if bestECombo.Error<bestNN1L.Error
        bestNN1L.etaInit=bestECombo.etaInit;
        bestNN1L.mu=bestECombo.mu;
        bestNN1L.alpha=bestECombo.alpha;
        bestNN1L.numUpdates=bestECombo.numUpdates;
        bestNN1L.Error=bestECombo.Error;
        bestNN1L.h=H;
    end
end
t2=datetime('now');
durGridNN1L=t2-t1; % Saves the duration of cross-validated grid search.



%%% 3. Learning curves against the data size %%%
% Initialize the best parameters %

conf.etaInit = bestNN1L.etaInit;
conf.numUpdates = bestNN1L.numUpdates;
conf.h = bestNN1L.h;
conf.alpha = bestNN1L.alpha;
conf.mu = bestNN1L.mu;
CV=10; % Default cross-validation
sizeinc= 1000; % NEEDS SETTING

t5=datetime('now');
[train_size, error_val, accuracy_val] = learningCurveNN(trainData, trainTarg, sizeinc, conf, CV);
t6=datetime('now');
durLearnCurve=t6-t5;

figure1=figure;
plot(train_size, accuracy_val)
title('Learning curves')
xlabel('Sample size')
ylabel('Classificaiton accuracy')
legend('Training data')
saveas(figure1,'learningcurveNN.jpg')

%%% 4. Fit NN model , once more on training data and best parameters %%%
conf.etaInit = bestNN1L.etaInit;
conf.numUpdates = bestNN1L.numUpdates;
conf.h = bestNN1L.h;
conf.alpha = bestNN1L.alpha;
conf.mu = bestNN1L.mu;
CV=10; % Default cross-validation

[errorCV, accuracyCV,tp,tn,fp,fn, score, target]=nnTrain1L(conf, CV, trainData, trainTarg); % MODEL

% Transform into nm x 1 dimensions
score=transpose(score);
score=score(:);

ypred=zeros(length(score),1);
for i=1:length(score)
    if score(i) > 0
        ypred(i)= 1;
    else
        ypred(i)=0;
    end
end

ytrue=zeros(length(trainTarg),1);
for i=1:length(trainTarg)
    if trainTarg(i) == -1
        ytrue(i)=0;
    else 
        ytrue(i)=1;
    end
end   
 

%%% 5. Confusion matrix and f1 Scores %%%
C = confusionmat(ytrue,ypred);
plotconfusion(transpose(ytrue), transpose(ypred));


%%% 6. Precision/Recall/F1 %%%
tp=mean(tp,2); % Mean true positive rate over all folds
fp=mean(fp,2); % Mean false positive rate over all folds
tn=mean(tn,2); % Mean true negative rate over all folds
fn=mean(fn,2); % Mean false negative rate over all folds

precision = tp/(tp+fp); % Precision for positve class
recall = tp/(tp+fn); % Recall for positive class

f1score=2*(precision*recall)/(precision+recall); % F1 score

%%% BONUS. Fit on the whole data %%%
[data, datatext, dataraw] = xlsread('C:\Users\Gedas\Desktop\City Uni\INM427. Neural computing\PROJECT\MATLAB\NNdatasetv2_SVM.xlsx');
trainDataFULL=StatisticalNormaliz(data(:, 3:end),'standard');
trainTargFULL=data(:, 2);

t7=datetime('now');
CV=3; % 3 Cross validation instead of 10 for quicker fit.
% Best parameters %
conf.etaInit = bestNN1L.etaInit;
conf.numUpdates = bestNN1L.numUpdates;
conf.h = bestNN1L.h;
conf.alpha = bestNN1L.alpha;
conf.mu = bestNN1L.mu;
% Best parameters %
[errorCVF, accuracyCVF,tpF,tnF,fpF,fnF, scoreF, targetF]=nnTrain1L(conf, CV, trainDataFULL, trainTargFULL); % MODEL
t8=datetime('now');
durFullTrain=t8-t7;


CF = confusionmat(trainTargFULL,labelF); % Confusion matrix 

tpF=mean(tpF,1); % Mean true positive rate over all folds
fpF=mean(fpF,1); % Mean false positive rate over all folds
tnF=mean(tnF,1); % Mean true negative rate over all folds
fnF=mean(fnF,1); % Mean false negative rate over all folds

precision = tpF/(tpF+fpF); % Precision for positve class
recall = tpF/(tpF+fnF); % Recall for positive class

f1score=2*(precision*recall)/(precision+recall); % F1 score


