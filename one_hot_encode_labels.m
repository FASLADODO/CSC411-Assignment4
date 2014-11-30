function [ labels ] = one_hot_encode_labels( training_labels, num_classes )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
num_training_examples = size(training_labels, 1);
labels = zeros(num_training_examples, num_classes);
for i = 1:num_training_examples
   labels(i,training_labels(i)) = 1; 
end

end

