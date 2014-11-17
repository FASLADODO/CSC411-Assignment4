function [ predictions ] = nn_classifier( tr_labels, tr_images, test_images, num_hiddens )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

inputs_train = reshape_data(tr_images);
inputs_test = reshape_data(test_images);
target_train = tr_labels;

num_inputs = size(inputs_train, 1);
num_outputs = 7;

W1 = 0.01 * randn(num_inputs, num_hiddens);
b1 = zeros(num_hiddens, 1);
W2 = 0.01 * randn(num_hiddens, num_outputs);
b2 = zeros(num_outputs, 1);

dW1 = zeros(size(W1));
dW2 = zeros(size(W2));
db1 = zeros(size(b1));
db2 = zeros(size(b2));

eps = 0.1;  %% the learning rate 
momentum = 0.0;   %% the momentum coefficient

num_epochs = 100; %% number of learning epochs (number of passes through the
                 %% training set) each time runbp is called.

total_epochs = 0; %% number of learning epochs so far. This is incremented 
                    %% by numEpochs each time runbp is called.

%%% For plotting learning curves:
min_epochs_per_plot = 200;
train_errors = zeros(1, min_epochs_per_plot);
epochs = [1 : min_epochs_per_plot];

for i = 1:10
    train_CE_list = zeros(1, num_epochs);

    start_epoch = total_epochs + 1;

    num_train_cases = size(inputs_train, 2);

    for epoch = 1:num_epochs
      % Fprop
      h_input = W1' * inputs_train + repmat(b1, 1, num_train_cases);  % Input to hidden layer.
      h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
      logit = W2' * h_output + repmat(b2, 1, num_train_cases);  % Input to output layer.
      prediction = 1 ./ (1 + exp(-logit));  % Output prediction.

      % Compute cross entropy
      train_CE = -mean(mean(target_train .* log(prediction) + (1 - target_train) .* log(1 - prediction)));

      % Compute deriv
      dEbydlogit = prediction - target_train;

      % Backprop
      dEbydh_output = W2 * dEbydlogit;
      dEbydh_input = dEbydh_output .* h_output .* (1 - h_output) ;

      % Gradients for weights and biases.
      dEbydW2 = h_output * dEbydlogit';
      dEbydb2 = sum(dEbydlogit, 2);
      dEbydW1 = inputs_train * dEbydh_input';
      dEbydb1 = sum(dEbydh_input, 2);

      %%%%% Update the weights at the end of the epoch %%%%%%
      dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1;
      dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2;
      db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1;
      db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2;

      W1 = W1 + dW1;
      W2 = W2 + dW2;
      b1 = b1 + db1;
      b2 = b2 + db2;
    end

    clf; 
    if total_epochs > min_epochs_per_plot
      epochs = [1 : total_epochs];
    end
end

end

