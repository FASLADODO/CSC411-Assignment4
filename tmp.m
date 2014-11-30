load labeled_images
load unlabeled_images
load public_test_images


%tr_images = reshape_data(tr_images);
test_images = reshape_data(public_test_images);
unlabeled_images = reshape_data(unlabeled_images);
inds = unidrnd(size(unlabeled_images, 2),1, 40000);

%test_images = gabor_features(test_images);
tr_images = gabor_features(tr_images);

pca_train_images = tr_images; %cat(2, tr_images, unlabeled_images(:,inds));


dims = size(tr_images, 1);
[base, mean, projX, eigval] = pcaimg(pca_train_images, dims);

plot(1:dims, eigval);

k = 1;
while (eigval(k) > 10)
    k = k + 1;
end

fprintf('Choose %d\n', k);