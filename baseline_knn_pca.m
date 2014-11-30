% The main function
% If test_images is provided, it will predict the results for those too, otherwise predicts 0 for the test cases.

load labeled_images.mat;
load public_test_images.mat;
%load hidden_test_images.mat;

% PCA
[coeff, score, latent] = princomp(tr_images);

% Transform test images using same eigenvalues - latent

% Keep x dimensions in both tr_images and test_images


h = size(tr_images,1);
w = size(tr_images,2);

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end

% Cross validation
for K=[3:10 15 20 35 50]
  nfold = 10;
  acc(K) = cross_validate_knn(K, tr_images, tr_labels, nfold);
  fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
end
[maxacc bestK] = max(acc);
fprintf('K is selected to be %d.\n', bestK);
% I get a bestK of 5

% Run the classifier
prediction = knn_classifier(bestK, tr_images, tr_labels, test_images);


% Fill in the test labels with 0 if necessary
generate_prediction_submission(prediction);

clear tr_images hidden_test_images public_test_images
