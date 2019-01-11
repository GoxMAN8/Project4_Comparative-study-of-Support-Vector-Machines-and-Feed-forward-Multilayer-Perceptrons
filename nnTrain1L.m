function [errorCV, accuracyCV, tp, tn, fp, fn, answer, target]= nnTrain1L(conf, CV, valData, valTarg)

%%%SINGLE LAYER FEEDFORWARD NEURAL NETWORK TRAINED WITH ERROR BACKPROPAGATION%%%
% Inputs %
% conf.etaInit -> learning rate
% conf.numUpdates -> maximum number of epochs
% conf.h -> number of hidden units
% conf.alpha -> learning rate decay factor
% conf.mu -> momentum factor
% CV -> cross validation

% Outputs %
% errorCV -> cross-validated error
% accuracyCV -> cross-validated accuracy. (1-errorCV)
% tp -> true positive cases
% tn -> true negative cases
% fp -> false positive cases
% fn -> false negative cases
% answer -> number of folds x (number of training examples/number of folds)
% matrix with networks output.
% target -> number of folds x (number of training examples/number of folds)
% target variables
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Transposing/Vectorizing data
trainData = transpose(valData);
trainTarg=transpose(valTarg);
originalInputSize = size(trainData,2);

                                                                                                                                                         
% Network Initialization %
K = 2; % Number of Layers
etaInit = conf.etaInit; % Learning Rate Initial Value
Delta = 0.1; % Stop Criterion #1
theta = 0.01; % Stop Criterion #2
numUpdates = conf.numUpdates; % Max number of epochs
N = size(trainData,2); % Number of Input Vectors
h = conf.h; % Number of Hidden Neurons
alpha = conf.alpha; % Learning Rate Decay Factor
mu = conf.mu; % Momentum constant;
M = CV; % Number of Folds (N = leave-one-out)
shuffle = 0; % Shuffle control (turned off for reproductibility)
classes = [1, -1];
E = 1;
% Network initialization done %
    
%fprintf('----------------------------\n');
%fprintf('- Neural Networks Training -\n');
%fprintf('----------------------------\n\n');

% CROSS VALIDATION INIT 
    
    if(M == N) % CONDITION FOR LEAVE ONE OUT CROSS VALIDATION
        N = size(trainData,2); 
        M = N;
    else
        N = size(trainData,2);
    end;  

%fprintf('Training type: %d folds on %d examples\n', M, originalInputSize);

[numAttr,numExamples] = size(trainData);

% Shuffling of data
    if (shuffle)
        trainData = [trainData ; trainTarg ; randn(1,numExamples)]';
        trainData = sortrows(trainData,numAttr+2)';
        trainTarg = trainData(numAttr+1,:); 
        trainData = trainData(1:numAttr,:);
    end;

% M-fold cross validation output storage init
answer = zeros(M, floor(N/M)); % Network output. (classes score)
target = zeros(M, floor(N/M)); % Output label. (classes label)

tp = zeros(1, M); % True positive instances
fp = zeros(1, M); % False positive instances
tn = zeros(1, M); % True negative instances
fn = zeros(1, M); % False negative instances
acc = zeros(1, M); % Accuracy

accuracyCV = 0; % Initialize cross validated accuracy

for m = 1:M
    if(m <= originalInputSize)
        %fprintf('Training fold %d/%d...\n', m, M);
    end;

    
    % Layers initialization 
    L(1).W = (1/sqrt(numExamples))*(rand(h,numAttr)-0.5); % Weight initialization from uniform random 
    L(1).b = (rand(h,1)-0.5)*0.2; % Bias initialization from uniform random. 
    L(2).W = (1/sqrt(conf.h))*(rand(1,h)-0.5); % Weight initialization from uniform random. 
    L(2).b = (rand(1,1)-0.5)*0.2; % Bias initialization from uniform random. 

    for k = 1:K 
        L(k).vb = zeros(size(L(k).b));
        L(k).vW = zeros(size(L(k).W));
    end;

    % Sequential Error Backpropagation Training
    n = 1; 
    i = 1; 
    finish = 0; 
    eta = etaInit;
    round = 1; 
    A(m,round) = 0;
    while not(finish)
        
        % Checking if it is a fold example
        ignoreTraining = 0; ignoreTesting = 0;
        if ((n > ((m-1)*floor(N/M))) && (n <= (m*floor(N/M))))
           ignoreTraining = 1;
           if((n > N) && ~shuffle)
               ignoreTesting = 1;
               break;
            end;
        end;
        
        J(m, i) = 0;
        if(~ignoreTraining)
            for(k = 1:K)
                L(k).db = zeros(size(L(k).b));
                L(k).dW = zeros(size(L(k).W));
            end;
        end;

        for ep = n:(n+E-1)
            
            if((ep > N) || ignoreTraining)
                break;
            end;
            
            % Feed-Forward         
            L(1).x = trainData(:,ep);
            for k = 1:K
                L(k).u = L(k).W * L(k).x + L(k).b;
                L(k).o = tanh(L(k).u);
                L(k+1).x = L(k).o;
            end; 
            e = trainTarg(n) - L(K).o;
            J(m,i) = J(m,i) + (e'*e)/2;

            % Error Backpropagation
            L(K+1).alpha = e; 
            L(K+1).W = eye(length(e));
            for k = fliplr(1:K)
                L(k).M = eye(length(L(k).o)) - diag(L(k).o)^2;
                L(k).alpha = L(k).M*L(k+1).W'*L(k+1).alpha;
                L(k).db = L(k).db + L(k).alpha;
                L(k).dW = L(k).dW + kron(L(k).x',L(k).alpha);
            end;
        end;

        % Updates
        for k = 1:K
            if(ignoreTraining)
                break; 
            end;
            L(k).vb = eta*L(k).db + mu*L(k).vb;
            L(k).b = L(k).b + L(k).vb;
            L(k).vW = eta*L(k).dW + mu*L(k).vW;
            L(k).W = L(k).W + L(k).vW;
        end;

        if(~ignoreTraining),
            A(m,round) = A(m,round) + (J(m, i)/(N-1));
            J(m, i) = J(m, i)/E;
        end;
            
        % Stop criterion delta/theta or maximum number of epochs
        if ((i > 1) && (n == N))
            if (((A(m,round) < Delta) && ((round > 2) && (abs(A(m,round-2)-A(m,round-1) < theta) && (abs(A(m,round-1)-A(m,round)) < theta)))) || (i > numUpdates))
                finish = 1;
            end;
        end;
        if not(finish)
            i = i+1; n = n+1; 
            if n > N 
                n = 1;
                round = round + 1;
                A(m,round) = 0;
            end; 
            eta = eta*alpha;
        end;

    end;

    % Test (test data)
    %fprintf('Testing fold %d/%d...\n', m, M);
    
    if(~ignoreTesting)
        index = 0;
        for n = ((m-1)*floor(N/M))+1:m*floor(N/M) % Looping over all small subset for testing      
            index = index + 1;
            L(1).x = trainData(:,n); % Input (first layer) equals test data.
            for k = 1:K % Feedforward the data through trained neural network
                L(k).u = L(k).W*L(k).x + L(k).b;
                L(k).o = tanh(L(k).u);
                L(k+1).x = L(k).o;
            end;
            answer(m,index) = L(K).o; % Store the networks' outputs. Dimension: folds x answer.
            target(m,index) = trainTarg(n); % Store the target variable . Dimension: folds x target

            if abs(answer(m,index) - target(m,index)) < (abs(classes(2) - classes(1))/2) % True if satisfied; False if not satisfied
                if(abs(answer(m,index)-classes(1)) < abs(answer(m,index)-classes(2))) 
                    tp(m) = tp(m) + 1; %True positive case
                else                     
                    tn(m) = tn(m) + 1; %True negative case
                end;
            else
                if(answer(m,index) > 0) %False positive case
                    fp(m) = fp(m) + 1;
                else                     %False negative case
                    fn(m) = fn(m) + 1;
                end;
            end;
        end;
    end;
        
    acc(m) = acc(m) + ((tp(m) + tn(m))/(tp(m) + tn(m) + fp(m) + fn(m)));
    accuracyCV = accuracyCV + (acc(m)/M);
    errorCV=1-accuracyCV;

end;
end

%fprintf('Average accuracy: %f\n\n', accuracyCV);
    