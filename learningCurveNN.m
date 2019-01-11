function [train_size, error_val, accuracy_val] = learningCurveNN(X, y, sizeinc, conf, CV)

%LEARNINGCURVE Generates the data size, training and cross validation set errors needed for the SVM model. 

%X is training data
%y is class labels
%sizeinc is size of epoch increments. 50 -> error rate will be calculated
%every 50 datapoints
% conf -> configuration of neural network

m = size(X, 1);

% Return and store error/epoch values correctly
error_val = zeros(m/sizeinc, 1);
accuracy_val   = zeros(m/sizeinc, 1);
train_size = zeros(m/sizeinc, 1);
indx=1; %Create indexing for data. 

% Loop over the training examples and calculate errors.
for i = sizeinc:sizeinc:m+1
    fprintf('Learning for %d epochs...\n', i)
    
    [errorCV, accuracyCV]=nnTrain1L(conf, CV, X(1:i, :), y(1:i)); % MODEL

   train_size(indx)=i; % Store the size of a training data
   error_val(indx)=errorCV;
   accuracy_val(indx)=accuracyCV;
   indx=indx+1; % Move of an index, of a data storage.
   fprintf('Learning finished for %d epchs.\n', i) 
   fprintf('------------------------------------------')
 
end
fprintf('---DONE---')
end