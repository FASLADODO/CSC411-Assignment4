load public_test_images
load labeled_images



tr_images = gabor_features(tr_images);
tr_images = tr_images';

tr_labels = one_hot_encode_labels(tr_labels, 7);
tr_labels = tr_labels';

k = 100;
i = 20;

performance = zeros(50);
[base, mean, tr_images, eigval] = pcaimg(tr_images, k);
for i = 1:50
    %change the values in this script to edit the parameters for the neural net
    hiddenLayerSize = i;
    neural_net_script_2;
    performance(i) = valPerformance;
end

i = find(performance == max(performance(:)));
hiddenLayerSize = i;
neural_net_script_2;

test_images = gabor_features(public_test_images);
test_images = test_images';
num_test_samples = size(test_images, 2);
test_images = test_images - repmat(mean, 1, num_test_samples);
proj_test = base'*test_images;
test_outputs = net(proj_test);

prediction = make_prediction_from_prob(test_outputs');
generate_prediction_submission(prediction);
