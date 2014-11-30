function [ predictions ] = multiclass_logistic_classifier( validationParam, tr_images, tr_labels, public_test_images)
%Trains a multiclass logistic classifier using the training data and labels
%provided, then predicts for the test data provided

    k = 7;                      %number of classes
    %t = 2000;                   %number of training iterations (only 1 should be uncommented)
    t = 4000;
    learning_rate = 0.006;      %learning rate
    %learning_rate = validationParam;

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

    for n = 1:t
        prob = logistic_regression_probabilities(weights, training_data);
        d_w = zeros(num_dimensions, k);
        err = prob-training_labels;
        for j = 1:num_dimensions
            training_data_dim = training_data(:,j)';
            d_w(j,:) = training_data_dim*err;
        end
        weights = weights - learning_rate*d_w;
        
        pred_prob = logistic_regression_probabilities(weights, test_data);
        predictions = zeros(num_test_examples,1);
        for i = 1:num_test_examples
            [tmp predictions(i)] = max(pred_prob(i,:));
        end
    end
end

