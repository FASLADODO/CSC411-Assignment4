load public_test_images
load labeled_images



tr_images = gabor_features(tr_images);
tr_images = tr_images';
tr_labels = one_hot_encode_labels(tr_labels, 7);
tr_labels = tr_labels';

%change the values in this script to edit the parameters for the neural net
neural_net_script_2;

test_images = gabor_features(public_test_images);
test_images = test_images';
test_outputs = net(test_images);
prediction = make_prediction_from_prob(test_outputs');
generate_prediction_submission(prediction);
