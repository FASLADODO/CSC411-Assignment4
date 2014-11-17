function [ input_targets ] = reshape_data( input_targets )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ntr = size(input_targets, 3);
h = size(input_targets,1);
w = size(input_targets,2);
input_targets = double(reshape(input_targets, [h*w, ntr]));

end

