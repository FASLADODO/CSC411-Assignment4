% Apply the PCA algorithm to the digit images. 
% Then plot the (sorted) eigenvalues and visualize the top-3 eigenvectors. 
%-------------------- Add your code here --------------------------------
[inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test] = load_data();
% [base,mean,projX,eigval] = pcaimg(inputs_train, 256);
% 
% plot(1:256, eigval)
% title('Plot of Sorted Eigen Values')
% ylabel('Eigen Value')
% pause
% 
% imagesc(reshape(base(:,1), 16, 16))
% colormap(gray)
% pause
% imagesc(reshape(base(:,2), 16, 16))
% pause
% imagesc(reshape(base(:,3), 16, 16))
% pause
% imagesc(reshape(mean(:,1), 16, 16))
% pause
% imagesc(reshape(mean(:,2), 16, 16))
% pause
% imagesc(reshape(mean(:,3), 16, 16))

% Conduct NN-based classification using PCA with different numbers of
% eigenvectors. Then show the
% error rate comparison.
%-------------------- Add your code here --------------------------------

m = [2, 5, 10, 20, 30, 50, 100];
validation_error = zeros(1,4);
test_error = zeros(1,7);
for i = 1:7
    [base, mean, projX, eigval] = pcaimg(inputs_train, m(i));
    
    validation_data = inputs_valid - repmat(mean, 1, 200);
    projValid = base'*validation_data;
    valid_predictions = run_knn(1, projX', target_train, projValid');
    validation_error(i) = classification_error(target_valid, valid_predictions);
    
    test_data = inputs_test - repmat(mean, 1, 400);
    projTest = base'*test_data;
    test_predictions = run_knn(1, projX', target_train, projTest');
    test_error(i) = classification_error(target_test, test_predictions);
end

plot(m, validation_error, m, test_error, 'LineWidth', 2)
title('Classification Rates vs Number of Eigen Vectors')
xlabel('Number of Eigen Vectors Kept')
ylabel('Classification Rate')
legend('Validation', 'Test')
pause
plot(m, 1-validation_error, m, 1-test_error, 'LineWidth', 2)
title('Error Rates vs Number of Eigen Vectors')
xlabel('Number of Eigen Vectors Kept')
ylabel('Error Rate')
legend('Validation', 'Test')