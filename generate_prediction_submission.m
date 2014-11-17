function [ ] = generate_prediction_submission( prediction )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
if (length(prediction) < 1253)
  prediction = [prediction; zeros(1253-length(prediction), 1)];
end


% Print the predictions to file

c = clock;
fix(c);
file_name = sprintf('prediction%d-%d-%d-%d-%d.csv', c(1), c(2), c(3), c(4), c(5));
sprintf('writing the output to %s\n', file_name)
fid = fopen(file_name, 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
  fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);

end

