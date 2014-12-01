load trainedmodel_multiclass;

% Assumes data is in public_test_images is being loaded from
% public_test_images and hidden_test_images.  If this is not the case, the
% value test_images needs to be set to a 32x32xn matrix, containing the
% test images, where n is the number of test examples.
load public_test_images;
load hidden_test_images;

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end


% Testing the model, prodcuing a nx1 matrix, where each element of the
% matrix is a value between 1 and K, where K is the number of classes.
test_data = reshape_data(test_images)';
test_data = test_data/1000;
test_data = test_data - (ones(num_test_examples, 1)*dimension_means);
pred_prob = logistic_regression_probabilities(weights, test_data);
predictions = zeros(num_test_examples,1);
for i = 1:num_test_examples
     [tmp predictions(i)] = max(pred_prob(i,:));
end

% predictions is the matrix that contains the predictions for the test
% data.