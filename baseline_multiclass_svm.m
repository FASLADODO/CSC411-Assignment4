load labeled_images.mat;
load public_test_images.mat;
%load hidden_test_images.mat;

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end

% Apply Gabor Energy Filters
test_images = gabor_features(test_images);
tr_images = gabor_features(tr_images);

SVMModel = fitcecoc(tr_images, tr_labels);
CVSVMModel = crossval(SVMModel);
kfoldLoss(CVSVMModel)