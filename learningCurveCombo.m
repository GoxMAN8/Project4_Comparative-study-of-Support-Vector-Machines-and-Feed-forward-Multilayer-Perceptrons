%%% LEARNING CURVES FOR BOTH ALGORITHMS %%%

figure1=figure;
plot(train_size,test_accuracy,train_size,accuracy_cv_NN, train_size, accuracy_cv_NN0)
title('Learning curves')
xlabel('Sample Size')
ylabel('Classification accuracy')
legend('CV SVM','BEST CV MLP', 'GRID CV MLP')
saveas(figure1,'learningcurvesCOMBO.jpg')