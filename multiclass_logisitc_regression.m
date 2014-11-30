%This scrip runs multiclass logistic regression using all the training data
%then tests it.  It should not be used, rather baseline_multiclass should
%be used as it includes validation steps.

load labeled_images.mat;
load public_test_images.mat;
k = 7;                  %number of classes
t = 100000;                 %number of training iterations
learning_rate = 0.006;

training_data = reshape_data(tr_images)';
training_data = training_data/1000;
test_data = reshape_data(public_test_images)';
test_data = test_data/1000;
num_training_examples = size(training_data, 1);
num_test_examples = size(test_data, 1);
num_dimensions = size(training_data, 2);

training_labels = zeros(num_training_examples, k);
for i = 1:num_training_examples
   training_labels(i,tr_labels(i)) = 1; 
end

dimension_means = mean(training_data, 1);
training_data = training_data - (ones(num_training_examples, 1)*dimension_means);
test_data = test_data - (ones(num_test_examples, 1)*dimension_means);

weights = randn(num_dimensions,k);


%train the model
for n = 1:t
    prob = logistic_regression_probabilities(weights, training_data);
    d_w = zeros(num_dimensions, k);
    err = prob-training_labels;
    for j = 1:num_dimensions
        training_data_dim = training_data(:,j)';
        d_w(j,:) = training_data_dim*err;
    end
    weights = weights - learning_rate*d_w;
end



%test the model
%compute the prediction probabilities
pred_prob = logistic_regression_probabilities(weights, test_data);
predictions = zeros(num_test_examples,1);
for i = 1:num_test_examples
    [tmp predictions(i)] = max(pred_prob(i,:));
end

generate_prediction_submission(predictions);