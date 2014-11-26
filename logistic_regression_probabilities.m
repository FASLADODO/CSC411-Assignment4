function [ prob ] = logistic_regression_probabilities( weights, inputs )
    [num_data_points num_dimensions] = size(inputs);
    sum_row = zeros(1, num_data_points);
    prob = exp(inputs*weights);
    for i = 1:num_data_points
        sum_row(i) = sum(prob(i,:));
    end
    prob = prob ./ (sum_row' * ones(1,7));


end

