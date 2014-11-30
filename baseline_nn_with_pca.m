load public_test_images
load labeled_images
load unlabeled_images

tr_images = reshape_data(tr_images);
pca_train_images = tr_images;
tr_labels = one_hot_encode_labels(tr_labels, 7);
tr_labels = tr_labels';
unlabeled_images = reshape_data(unlabeled_images);
inds = unidrnd(size(unlabeled_images, 2),1, 40000);

pca_train_images = cat(2, pca_train_images, unlabeled_images(:,inds));

dims = 650;
[base, mean, proj, eigval] = pcaimg(pca_train_images, dims);

tr = tr_images - repmat(mean, 1, size(tr_images, 2));
tr_images = base'*tr;
%change the values in this script to edit the parameters for the neural net
hiddenLayerSize = 100;
neural_net_script_2;

test_images = reshape_data(public_test_images);
num_test_samples = size(test_images, 2);
test_images = test_images - repmat(mean, 1, num_test_samples);
proj_test = base'*test_images;
test_outputs = net(proj_test);
prediction = make_prediction_from_prob(test_outputs');
generate_prediction_submission(prediction);
