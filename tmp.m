
predictions = zeros(size(test_outputs, 1),1);
for i = 1:size(test_outputs,1)
    [tmp predictions(i)] = max(test_outputs(i,:));
end