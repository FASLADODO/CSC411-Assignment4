function [ predictions ] = make_prediction_from_prob( prob_inputs )

length = size(prob_inputs, 1);
predictions = zeros(length,1);
for i = 1:length
    [tmp predictions(i)] = max(prob_inputs(i,:));
end


end

