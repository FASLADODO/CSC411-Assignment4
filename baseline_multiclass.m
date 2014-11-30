load labeled_images.mat;
load public_test_images.mat;
%load hidden_test_images.mat;

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end


% k represents either learning rate or number of iterations, changed in the
% multiclass_logistic_classifier function
K = 1;
diff = 1;
while (K < 4) || (acc(K-diff) > acc(K-2*diff) || (acc(K-diff) > acc(K-3*diff)))
  nfold = 5;
  acc(K) = cross_validate_multi_logistic(K*100, tr_images, tr_labels, nfold);
  fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K*100, acc(K));
  K = K + diff;
end

plot(K, acc);
[maxacc bestK] = max(acc);
fprintf('K is selected to be %d.\n', bestK);

prediction = multiclass_logistic_classifier(bestK, tr_images, tr_labels, test_images);
generate_prediction_submission(prediction);